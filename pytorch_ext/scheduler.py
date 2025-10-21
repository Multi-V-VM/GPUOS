import contextlib
import threading
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch

try:
    import gpuos_ext  # built extension (see examples for dynamic build)
except Exception as e:  # pragma: no cover
    gpuos_ext = None


class _GPUOSSchedulerMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, *, size_threshold: int, auto_flush_ms: Optional[float], pending: Set[int]):
        super().__init__()
        self.size_threshold = size_threshold
        self.auto_flush_ms = auto_flush_ms
        self.pending = pending  # set of storage data_ptr ints for pending outputs
        self.fused = {}  # data_ptr(int) -> node dict: {'arity','expr','a','b','shape','dtype'}
        self._stop = False
        self._bg: Optional[threading.Thread] = None
        if auto_flush_ms and auto_flush_ms > 0:
            self._bg = threading.Thread(target=self._flusher, daemon=True)
            self._bg.start()

    def close(self):
        self._stop = True
        if self._bg:
            self._bg.join(timeout=0.1)

    def _flusher(self):
        # Periodically flush pending requests; best-effort
        while not self._stop:
            time.sleep(self.auto_flush_ms / 1000.0)
            try:
                # Flush fused nodes first
                self.flush_fused()
                if gpuos_ext is not None:
                    gpuos_ext.flush(sync=False)
            except Exception:
                pass

    def _maybe_flush_on_dependency(self, func, args, kwargs):
        # If any arg is a Tensor that corresponds to a pending output, flush synchronously
        # Skip for ops we intend to fuse
        fusible = {
            'aten::add.Tensor', 'aten::sub.Tensor', 'aten::mul.Tensor', 'aten::div.Tensor',
            'aten::maximum', 'aten::minimum', 'aten::pow.Tensor_Tensor',
            'aten::relu', 'aten::sigmoid', 'aten::tanh', 'aten::exp', 'aten::log', 'aten::sqrt', 'aten::abs',
            'aten::hardsigmoid', 'aten::hardswish', 'aten::gelu', 'aten::sin', 'aten::cos',
            'aten::leaky_relu', 'aten::hardtanh', 'aten::elu', 'aten::softplus', 'aten::clamp', 'aten::clamp_min', 'aten::clamp_max',
            'aten::sum.dim_IntList', 'aten::mean.dim',
        }
        if func.name() in fusible:
            return
        def _check_tensor(x: Any) -> bool:
            if not isinstance(x, torch.Tensor):
                return False
            if not x.is_cuda:
                return False
            try:
                ptr = x.data_ptr()
            except Exception:
                return False
            return ptr in self.pending

        def _walk(o: Any) -> bool:
            if _check_tensor(o):
                return True
            if isinstance(o, (list, tuple)):
                return any(_walk(i) for i in o)
            if isinstance(o, dict):
                return any(_walk(v) for v in o.values())
            return False

        if _walk(args) or _walk(kwargs or {}):
            if gpuos_ext is not None:
                self.flush_fused()
                gpuos_ext.flush(sync=True)
            # clear all pending markers; conservative
            self.pending.clear()
            self.fused.clear()

    def _try_schedule_binary(self, func, a: torch.Tensor, b: torch.Tensor, *, alpha=None, out: Optional[torch.Tensor] = None):
        # Fusion-first path for binary elementwise
        if torch.is_grad_enabled():
            return None
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return None
        if not (a.is_cuda and b.is_cuda):
            return None
        if a.numel() > self.size_threshold or b.numel() > self.size_threshold:
            return None

        # Prepare out
        if out is None:
            try:
                shape = torch.broadcast_shapes(tuple(a.shape), tuple(b.shape))
            except Exception:
                return None
            dtype = torch.result_type(a, b)
            out = torch.empty(shape, dtype=dtype, device=a.device)
        else:
            if not out.is_cuda or out.device != a.device:
                return None

        name = func.name()
        elementwise_map = {
            'aten::add.Tensor':         '(A + B)',
            'aten::sub.Tensor':         '(A - B)',
            'aten::mul.Tensor':         '(A * B)',
            'aten::div.Tensor':         '(A / B)',
            'aten::maximum':            '(A > B ? A : B)',
            'aten::minimum':            '(A < B ? A : B)',
            'aten::pow.Tensor_Tensor':  'powf(A, B)',
        }
        if name not in elementwise_map:
            return None
        expr = elementwise_map[name]
        node = {'arity': 2, 'expr': expr, 'a': a, 'b': b, 'shape': tuple(out.shape), 'dtype': out.dtype}
        self.fused[out.data_ptr()] = node
        self.pending.add(out.data_ptr())
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Flush if downstream dependency uses any pending outputs for non-fusible ops
        self._maybe_flush_on_dependency(func, args, kwargs or {})

        name = func.name()
        if name in (
            'aten::add.Tensor', 'aten::sub.Tensor', 'aten::mul.Tensor', 'aten::div.Tensor',
            'aten::maximum', 'aten::minimum', 'aten::pow.Tensor_Tensor'
        ):
            # Normalize args for (a, b, alpha?, out?) forms
            # aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
            # aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
            # We only handle .Tensor overload and the function we see here is that exact overload name.
            a, b = args[0], args[1]
            alpha = kwargs.get('alpha', None) if kwargs else None
            out = kwargs.get('out', None) if kwargs else None
            scheduled = self._try_schedule_binary(func, a, b, alpha=alpha, out=out)
            if scheduled is not None:
                return scheduled
        elif name in ('aten::relu', 'aten::sigmoid', 'aten::tanh', 'aten::exp', 'aten::log', 'aten::sqrt', 'aten::abs',
                      'aten::hardsigmoid', 'aten::hardswish', 'aten::gelu', 'aten::sin', 'aten::cos'):
            x = args[0]
            if torch.is_grad_enabled() or not x.is_cuda or x.numel() > self.size_threshold:
                return func(*args, **(kwargs or {}))
            out = (kwargs or {}).get('out', None)
            if out is None:
                out = torch.empty_like(x)
            u_map = {
                'aten::relu':   ('(A > 0.f ? A : 0.f)', 1),
                'aten::sigmoid':('1.f / (1.f + expf(-A))', 1),
                'aten::tanh':   ('tanhf(A)', 1),
                'aten::exp':    ('expf(A)', 1),
                'aten::log':    ('logf(A)', 1),
                'aten::sqrt':   ('sqrtf(A)', 1),
                'aten::abs':    ('fabsf(A)', 1),
                'aten::hardsigmoid': ('fminf(fmaxf(0.2f*A + 0.5f, 0.f), 1.f)', 1),
                'aten::hardswish':  ('A * fminf(fmaxf(A/6.f + 0.5f, 0.f), 1.f)', 1),
                'aten::gelu':   ('0.5f * A * (1.f + tanhf(0.79788456f*(A + 0.044715f*A*A*A)))', 1),
                'aten::sin':    ('sinf(A)', 1),
                'aten::cos':    ('cosf(A)', 1),
            }
            expr, arity = u_map[name]
            # Fuse on top of existing fused node (unary-after-*) if possible
            base_ptr = None
            try:
                base_ptr = x.data_ptr()
            except Exception:
                base_ptr = None
            if base_ptr is not None and base_ptr in self.fused:
                base = self.fused.pop(base_ptr)
                # Compose unary(expr(base))
                new_expr = expr.replace('A', f"({base['expr']})")
                node = {'arity': base['arity'], 'expr': new_expr, 'a': base['a'], 'b': base.get('b'), 'shape': base['shape'], 'dtype': base['dtype']}
                out = torch.empty(node['shape'], dtype=node['dtype'], device=x.device)
                self.fused[out.data_ptr()] = node
                self.pending.discard(base_ptr)
                self.pending.add(out.data_ptr())
                return out
            else:
                node = {'arity': 1, 'expr': expr, 'a': x, 'b': None, 'shape': tuple(out.shape), 'dtype': out.dtype}
                self.fused[out.data_ptr()] = node
                self.pending.add(out.data_ptr())
                return out

        elif name in ('aten::leaky_relu', 'aten::hardtanh', 'aten::elu', 'aten::softplus', 'aten::clamp', 'aten::clamp_min', 'aten::clamp_max'):
            x = args[0]
            if torch.is_grad_enabled() or not x.is_cuda or x.numel() > self.size_threshold:
                return func(*args, **(kwargs or {}))
            out = (kwargs or {}).get('out', None)
            if out is None:
                out = torch.empty_like(x)

            def f32(v, default):
                if v is None:
                    v = default
                return float(v)

            if name == 'aten::leaky_relu':
                ns = f32((kwargs or {}).get('negative_slope', args[1] if len(args) > 1 else 0.01), 0.01)
                expr = f'(A >= 0.f ? A : {ns:.8f}f*A)'
                key = f'leaky_relu|f32|{ns:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::hardtanh':
                mn = f32((kwargs or {}).get('min_val', args[1] if len(args) > 1 else -1.0), -1.0)
                mx = f32((kwargs or {}).get('max_val', args[2] if len(args) > 2 else 1.0), 1.0)
                expr = f'fminf(fmaxf(A, {mn:.8f}f), {mx:.8f}f)'
                key = f'hardtanh|f32|{mn:.8f}|{mx:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::elu':
                alpha = f32((kwargs or {}).get('alpha', args[1] if len(args) > 1 else 1.0), 1.0)
                expr = f'(A > 0.f ? A : {alpha:.8f}f*(expf(A) - 1.f))'
                key = f'elu|f32|{alpha:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::softplus':
                beta = f32((kwargs or {}).get('beta', args[1] if len(args) > 1 else 1.0), 1.0)
                expr = f'(log1pf(expf({beta:.8f}f*A))/{beta:.8f}f)'
                key = f'softplus|f32|{beta:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::clamp':
                mn = (kwargs or {}).get('min', None)
                mx = (kwargs or {}).get('max', None)
                if mn is None or mx is None:
                    return func(*args, **(kwargs or {}))
                mn = f32(mn, 0.0); mx = f32(mx, 0.0)
                expr = f'fminf(fmaxf(A, {mn:.8f}f), {mx:.8f}f)'
                key = f'clamp|f32|{mn:.8f}|{mx:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::clamp_min':
                mn = f32((kwargs or {}).get('min', args[1] if len(args) > 1 else 0.0), 0.0)
                expr = f'fmaxf(A, {mn:.8f}f)'
                key = f'clamp_min|f32|{mn:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            elif name == 'aten::clamp_max':
                mx = f32((kwargs or {}).get('max', args[1] if len(args) > 1 else 0.0), 0.0)
                expr = f'fminf(A, {mx:.8f}f)'
                key = f'clamp_max|f32|{mx:.8f}'
                slot = gpuos_ext.register_elementwise(key, expr, 1)
                gpuos_ext.submit_unary(slot, x, out)
            else:
                return func(*args, **(kwargs or {}))
            self.pending.add(out.data_ptr())
            return out

    # Mode-scoped helpers for fused nodes
    def _register_submit_node(self, node):
        expr = node['expr']; arity = node['arity']; a = node['a']; b = node.get('b')
        key = f"fused|{arity}|{expr}"
        slot = gpuos_ext.register_elementwise(key, expr, arity)
        out = torch.empty(node['shape'], dtype=node['dtype'], device=a.device)
        if arity == 1:
            gpuos_ext.submit_unary(slot, a, out)
        else:
            gpuos_ext.submit_binary(slot, a, b, out)
        return out

    def flush_fused(self):
        if not self.fused:
            return
        items = list(self.fused.items())
        self.fused.clear()
        for _, node in items:
            try:
                self._register_submit_node(node)
            except Exception:
                pass

        # Reductions: multi-dim sum/mean (rrank>=1), generic; we currently reduce synchronous only when dims provided
        elif name in ('aten::sum.dim_IntList', 'aten::mean.dim'):
            x = args[0]
            if torch.is_grad_enabled() or not x.is_cuda or x.numel() > self.size_threshold:
                return func(*args, **(kwargs or {}))
            # Extract dim and keepdim
            dim = None
            keepdim = bool((kwargs or {}).get('keepdim', False))
            if 'dim' in (kwargs or {}):
                dim = (kwargs or {})['dim']
            elif len(args) > 1:
                dim = args[1]
            if dim is None:
                return func(*args, **(kwargs or {}))
            # Normalize dims to list of sorted unique positive axes
            if isinstance(dim, int):
                dims = [dim]
            elif isinstance(dim, (list, tuple)):
                dims = list(dim)
            else:
                return func(*args, **(kwargs or {}))
            nd = x.dim()
            ndims = []
            for d in dims:
                d = int(d)
                if d < 0:
                    d += nd
                if d < 0 or d >= nd:
                    return func(*args, **(kwargs or {}))
                ndims.append(d)
            ndims = sorted(set(ndims))
            if len(ndims) == 0:
                return x.clone()  # nothing to reduce
            # Output shape
            if keepdim:
                out_shape = list(x.shape)
                for d in ndims:
                    out_shape[d] = 1
            else:
                out_shape = [x.shape[i] for i in range(nd) if i not in set(ndims)]
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            op_name = 'mean' if name == 'aten::mean.dim' else 'sum'
            slot = gpuos_ext.register_reduce(f"reduce_{op_name}|axes={ndims}|keep={int(keepdim)}|{str(x.dtype)}", op_name)
            gpuos_ext.submit_reduce(slot, x, out, ndims, keepdim)
            self.pending.add(out.data_ptr())
            return out

        # Fallback to default behavior
        return func(*args, **(kwargs or {}))


class GPUOSScheduler:
    def __init__(self, *, capacity: int = 8192, threads_per_block: int = 256, size_threshold: int = 1 << 16, auto_flush_ms: Optional[float] = 2.0):
        if gpuos_ext is None:
            raise RuntimeError('gpuos_ext not built/available; see examples/pytorch_batch_demo.py for dynamic build usage.')
        self.capacity = capacity
        self.threads_per_block = threads_per_block
        self.size_threshold = size_threshold
        self.auto_flush_ms = auto_flush_ms
        self._mode: Optional[_GPUOSSchedulerMode] = None
        self._pending: Set[int] = set()

    def __enter__(self):
        gpuos_ext.init(self.capacity, self.threads_per_block)
        self._mode = _GPUOSSchedulerMode(size_threshold=self.size_threshold, auto_flush_ms=self.auto_flush_ms, pending=self._pending)
        self._cm = self._mode.__enter__()
        return self

    def flush(self, sync: bool = False):
        # Flush fused elementwise nodes
        if self._mode is not None:
            self._mode.flush_fused()
        # Flush extension-side pending batch (if any)
        if gpuos_ext is not None:
            gpuos_ext.flush(sync=sync)
        if sync:
            self._pending.clear()

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._mode is not None:
                self._mode.close()
                self._mode.__exit__(exc_type, exc, tb)
        finally:
            if self._mode is not None:
                self._mode.flush_fused()
            gpuos_ext.flush(sync=True)
            self._pending.clear()
            gpuos_ext.shutdown()

    # Mode-scoped helpers for flush: perform registration + submit for fused nodes
    def _register_submit_node(self, node):
        expr = node['expr']; arity = node['arity']; a = node['a']; b = node.get('b');
        key = f"fused|{arity}|{expr}"
        slot = gpuos_ext.register_elementwise(key, expr, arity)
        out = torch.empty(node['shape'], dtype=node['dtype'], device=a.device)
        if arity == 1:
            gpuos_ext.submit_unary(slot, a, out)
        else:
            gpuos_ext.submit_binary(slot, a, b, out)
        return out

    def flush_fused(self):
        if not self.fused:
            return
        items = list(self.fused.items())
        self.fused.clear()
        for ptr, node in items:
            try:
                self._register_submit_node(node)
            except Exception:
                pass


@contextlib.contextmanager
def scheduler_context(*, capacity: int = 8192, threads_per_block: int = 256, size_threshold: int = 1 << 16, auto_flush_ms: Optional[float] = 2.0):
    sched = GPUOSScheduler(capacity=capacity, threads_per_block=threads_per_block, size_threshold=size_threshold, auto_flush_ms=auto_flush_ms)
    with sched:
        yield sched
