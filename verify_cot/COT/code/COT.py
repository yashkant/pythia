import torch
import torch.nn as nn
import torch.nn.functional as F

# For CIFAR10 we difine classes to 10
classes = 10

# Complement Entropy (CE)


class ComplementEntropy(nn.Module):

    def __init__(self):
        super(ComplementEntropy, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = classes
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss


class ComplementEntropyCustom(nn.Module):

    def __init__(self):
        super(ComplementEntropyCustom, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, x, y):
        """ Returns the complement entropy loss as proposed in the report.

            This implementation is faithful with Equation (6) in the report's
            section (2.4).

            Equation (6) talks about the complement entropy, we calculate the
            negative of its value i.e. complement entropy loss.
        """
        # ----------------------------------------------------------------------
        # Convert to one-hot. Apply softmax to predictions.
        # ----------------------------------------------------------------------
        y_onehot = torch.FloatTensor(x.shape).zero_()
        y_onehot.scatter_(1, torch.unsqueeze(y, 1).data.cpu(), 1)
        y = y_onehot.cuda()
        x = F.softmax(x, dim=1)

        # ----------------------------------------------------------------------
        # Negated complement entropy loss for each label with zero target score
        # ----------------------------------------------------------------------

        y_is_0 = torch.eq(y, 0)
        x_remove_0 = x.clone().masked_fill_(y_is_0, 0)
        xr_sum = torch.sum(x_remove_0, dim=1, keepdim=True)
        one_min_xr_sum = 1 - xr_sum
        one_min_xr_sum.masked_fill_(one_min_xr_sum <= 0,
                                    1e-7)  # Numerical issues
        px = x / one_min_xr_sum
        log_px = torch.log(px + 1e-10)  # Numerical issues
        new_x = px * log_px
        loss = new_x * (y_is_0.float())  # Remove non-zero labels loss

        # ----------------------------------------------------------------------
        # Normalize the loss to balance it with cross entropy loss
        # ----------------------------------------------------------------------
        num_labels = y.size()[1]
        batch_size = y.size()[0]
        normalize = 1 / (num_labels*batch_size)
        loss = loss * normalize
        loss_return = torch.sum(loss)

        return loss_return
