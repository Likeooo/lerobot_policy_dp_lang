def check_stride_mismatch(model):
    print(">>> 开始注册梯度检查 Hook...")
    
    # 遍历模型所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 使用闭包绑定参数名
            def get_hook(name_p):
                def hook(grad):
                    if grad is not None:
                        # 填入报错信息中的数值进行匹配
                        target_shape = [1, 32, 1, 1]
                        target_stride = [32, 1, 32, 32] # 报错里的 grad.strides()
                        
                        # 检查形状是否匹配
                        if list(grad.shape) == target_shape:
                            # 检查 Stride 是否异常 (或者直接匹配报错的 stride)
                            if list(grad.stride()) == target_stride:
                                print(f"\n[!!! 抓到了 !!!] 参数名称: {name_p}")
                                print(f"    Shape: {grad.shape}")
                                print(f"    Stride: {grad.stride()}")
                                print(f"    Is_contiguous: {grad.is_contiguous()}")
                return hook
            
            param.register_hook(get_hook(name))