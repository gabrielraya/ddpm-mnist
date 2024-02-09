"""
All functions related to loss computation and optimization.
"""
import torch


def get_model_fn(model, train=False):
    """
    Returns a function that runs the model in either training or evaluation mode.
    """
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_ddpm_loss_fn(diffusion, train=True):
    """ Implements DDPM loss """
    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train) # set model either in train of evaluation mode
        time_steps = torch.randint(0, diffusion.T, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = diffusion.sqrt_1m_alphas_cumprod.to(batch.device)

        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[time_steps, None, None, None] * batch + sqrt_1m_alphas_cumprod[time_steps, None, None, None] * noise

        predicted_noise = model_fn(perturbed_data, time_steps)

        losses = torch.square(predicted_noise - noise)
        loss = torch.mean(losses)
        return loss
    return loss_fn


def get_step_fn(diffusion, train=True, optimize_fn=None):
    """
    Create a one-step function for training or evaluation.
    """
    loss_fn = get_ddpm_loss_fn(diffusion, train)

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        Args:
            state: A dictionary of training information, containing the score model,
                    optimizer, EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']

        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
        return loss

    return step_fn