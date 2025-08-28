import torch
import torch.nn.functional as F

class MyMixUp:
    def __init__(self, alpha=10, seed=42):
        self.alpha = alpha
        self.seed = seed
        self.set_random_seed()

    def set_random_seed(self):
        torch.manual_seed(self.seed)

    def mix(self, inputs, labels):
        # Perform the MixUp data augmentation technique on a batch of inputs and labels.
        #
        # Parameters:
        # inputs (torch.Tensor): A batch of input data.
        # labels (torch.Tensor): A batch of labels.
        #
        # Returns:
        # Tuple[torch.Tensor, torch.Tensor]:
        # mixed_inputs (torch.Tensor): The mixed input data (images which mixed together).
        # mixed_labels (torch.Tensor): The mixed labels in one-hot encoded format.
        #
        # Usage
        # Use to prepare mixed data for train model 2 (ELM + MIXUP) and 4 (ELM + MIXUP + ENSEMBLE)
        # And use to generate mixup.png

        batch_size = inputs.size(0) # 40

        indices = torch.randperm(batch_size)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        mixed_inputs = lam * inputs + (1 - lam) * inputs[indices]

        labels_one_hot = F.one_hot(labels, num_classes=10).float()
        shuffled_one_hot = F.one_hot(labels[indices], num_classes=10).float()

        mixed_labels = lam * labels_one_hot + (1 - lam) * shuffled_one_hot

        return mixed_inputs, mixed_labels
    
    def soft_cross_entropy(self, predicts, targets):
        log_probs = F.log_softmax(predicts, dim=1)
        loss = -1 * ((targets * log_probs).sum(dim=1).mean())

        return loss
