# 接收参数为形状为torch.Size([1, 192, 80, 80])的一个张量
# 实现对这个张量的Grad-CAM算法
import torch
import torch.nn.functional as F

def grad_cam(input_tensor, model, target_layer):
    """
    Compute Grad-CAM for a given input tensor and model.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [1, 192, 80, 80].
        model (torch.nn.Module): The model to use for Grad-CAM.
        target_layer (torch.nn.Module): The target layer to compute Grad-CAM for.

    Returns:
        torch.Tensor: Grad-CAM heatmap.
    """
    # Forward pass
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()

    # Assume the target class is the one with the highest score
    target_class = output.argmax(dim=1).item()
    target = output[0, target_class]

    # Backward pass
    target.backward()

    # Get activations and gradients
    activations = activations[0].detach()
    gradients = gradients[0].detach()

    # Compute Grad-CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    grad_cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

    # Normalize Grad-CAM
    grad_cam = F.interpolate(grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return grad_cam