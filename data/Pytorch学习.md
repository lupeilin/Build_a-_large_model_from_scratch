## `nn.Parameter` 

是 PyTorch 中用于**将张量注册为模型的可训练参数**的核心类。简单来说：当一个张量被包装成 `nn.Parameter` 后，它会自动被添加到 `nn.Module` 的参数列表中，从而能够在训练过程中被优化器更新。

---

### 一、基本用法

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 普通张量不会被自动注册
        self.w = torch.randn(3, 3)          # 不是 Parameter，不会被优化器更新
        # 使用 nn.Parameter 注册
        self.bias = nn.Parameter(torch.zeros(3))  # 可训练参数
```

- `nn.Parameter` 是 `torch.Tensor` 的子类，所以它拥有张量的所有属性和方法。
- 当被赋值给 `nn.Module` 的属性时，会自动调用 `module.register_parameter()`，将该参数加入模块的 `parameters()` 迭代器中。

---

### 二、与普通张量的区别

| 特性                           | 普通 `torch.Tensor` | `nn.Parameter`                 |
| ------------------------------ | ------------------- | ------------------------------ |
| 自动注册为模型参数             | ❌                   | ✅                              |
| 被 `module.parameters()` 包含  | ❌                   | ✅                              |
| 优化器默认更新                 | ❌                   | ✅（如果 `requires_grad=True`） |
| 保存和加载模型（`state_dict`） | ❌                   | ✅                              |
| 可移动设备（`.to(device)`）    | 手动                | 随模型自动移动                 |

**关键**：如果只是把张量作为模型的成员变量，优化器将无法看到它，也就不会更新其值。必须用 `nn.Parameter` 包装（或者使用 `nn.ParameterList` / `nn.ParameterDict` 等容器）。

---

### 三、工作原理

当你定义 `self.bias = nn.Parameter(torch.zeros(3))` 时：
1. `__setattr__` 方法检测到赋值的是 `nn.Parameter` 实例，将其存入模块内部的 `_parameters` 字典。
2. `module.parameters()` 会递归遍历所有 `_parameters` 中的项。
3. 调用 `module.to(device)` 时，参数会自动移动到目标设备。
4. 调用 `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)` 时，优化器会获得这些参数的引用，并在反向传播后更新它们。

---

### 四、常见示例

### 1. 自定义线性层（无偏置）
```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    def forward(self, x):
        return x @ self.weight.T
```

### 2. 可训练的缩放因子（如 LayerNorm 中的 `scale` 和 `shift`）
```python
class MyLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        # 假设 x 形状 (..., dim)
        return self.scale * x + self.shift
```

### 3. 可训练的位置编码
```python
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(max_len, emb_dim))
    def forward(self, x):
        return x + self.pos_emb[:x.size(1)]
```

---

### 五、注意事项

1. **`requires_grad` 默认是 `True`**  
   `nn.Parameter(torch.ones(5))` 等价于 `nn.Parameter(torch.ones(5), requires_grad=True)`。如果不想训练该参数（例如冻结某层），可以在创建后设置 `param.requires_grad = False`。

2. **不要直接修改参数数据**  
   应使用 `param.data` 进行原地操作，否则可能破坏梯度计算图。例如：
   ```python
   # 推荐
   param.data.fill_(0)
   # 不推荐（会改变 requires_grad 属性等）
   param.fill_(0)
   ```

3. **参数与缓冲区（buffer）的区别**  
   - 参数（Parameter）：需要被优化器更新，如权重、偏置。
   - 缓冲区（buffer）：不需要训练但需要保存的状态，如 BatchNorm 的 running_mean。通过 `register_buffer` 注册。

4. **保存和加载**  
   用 `torch.save(model.state_dict(), 'model.pth')` 时，所有 `nn.Parameter` 会自动保存。

---

### 六、总结

`nn.Parameter` 是 PyTorch 中定义可训练参数的标准方法。它将普通张量“升级”为模型参数，使其能够被优化器更新、被 `state_dict` 保存、并随模型自动移动设备。在实现自定义层时，凡是要让优化器更新的权重、偏置等，都应该用 `nn.Parameter` 包装。