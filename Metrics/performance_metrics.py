import torch
import torchmetrics
# .classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import numpy as np
import torch.nn.functional as F



# def split_targets(t, num_classes, pos_class_val=255):
#     out = []
#     for i in range(num_classes):
#         out.append(torch.where(t == i, pos_class_val, 0))

#     return(out)

# def process_results(target, pred, clss_num):
#     pred = pred.reshape(-1)
#     target = target.reshape(-1)

#     # subtracts class_num from the values in pred except for 0 values
#     if clss_num != 0:
#         pred = torch.where(pred == 0, 0, pred - clss_num)


#     return(target, pred, clss_num)

class ProcessClasses(torch.nn.Module):
    def __init__(self):
        super(ProcessClasses, self).__init__()

    def forward(self, probs, targets, child_classes=False):
        if child_classes:
            # Shift classes 0-3 to 1-4 and assign background class (0) to pixels without any positive indication
            # Add background class (0) at the start
            background_class = (torch.sum(probs, dim=1, keepdim=True) == 0).float()
            probs = torch.cat([background_class, probs], dim=1)
            targets_background_class = (torch.sum(targets, dim=1, keepdim=True) == 0).float()
            targets = torch.cat([targets_background_class, targets], dim=1)

            # Convert to class indices
            probs = torch.argmax(probs, dim=1).float()
            targets = torch.argmax(targets, dim=1).float()
        else:
            probs = torch.argmax(probs, dim=1).float()
            targets = torch.argmax(targets, dim=1).float()

        return probs, targets



# TODO: change both to multi calss metrics
class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.process_classes = ProcessClasses()
        # self.smooth = smooth

    def forward(self, probs, targets, device, num_classes, child_classes=False):
        probs, targets = self.process_classes(probs, targets, child_classes)
        # Calculate the metric for each class
        if child_classes:
            metric2 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes+1, average=None, ignore_index=0).to(device)
            return(metric2(probs, targets)[1:])
        else:
            metric2 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=None, ignore_index=-1).to(device)
            return (metric2(probs, targets))
        




class Jaccardindex(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Jaccardindex, self).__init__()
        self.process_classes = ProcessClasses()
        # self.smooth = smooth

    def forward(self, probs, targets, device, num_classes, child_classes=False):
        probs, targets = self.process_classes(probs, targets, child_classes)
        # Calculate the metric for each class
        if child_classes:
            metric2 = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes+1, average=None, ignore_index=0).to(device)
            return(metric2(probs, targets)[1:])
        else:
            metric2 = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes, average=None, ignore_index=-1).to(device)
            return (metric2(probs, targets))
    


class Accuracy(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Accuracy, self).__init__()
        self.process_classes = ProcessClasses()
        # self.smooth = smooth

    def forward(self, probs, targets, device, num_classes, child_classes=False):
        probs, targets = self.process_classes(probs, targets, child_classes)
        # Calculate the metric for each class
        if child_classes:
            metric2 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes+1, average=None, ignore_index=0).to(device)
            return(metric2(probs, targets)[1:])
        else:
            metric2 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average=None, ignore_index=-1).to(device)
            return (metric2(probs, targets))
        



class Precision(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Precision, self).__init__()
        self.process_classes = ProcessClasses()
        # self.smooth = smooth

    def forward(self, probs, targets, device, num_classes, child_classes=False):
        probs, targets = self.process_classes(probs, targets, child_classes)
        # Calculate the metric for each class
        if child_classes:
            metric2 = torchmetrics.Precision(task='multiclass', num_classes=num_classes+1, average=None, ignore_index=0).to(device)
            return(metric2(probs, targets)[1:])
        else:
            metric2 = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average=None, ignore_index=-1).to(device)
            return (metric2(probs, targets))
        


class Recall(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Recall, self).__init__()
        self.process_classes = ProcessClasses()
        # self.smooth = smooth

    def forward(self, probs, targets, device, num_classes, child_classes=False):
        probs, targets = self.process_classes(probs, targets, child_classes)
        # Calculate the metric for each class
        if child_classes:
            metric2 = torchmetrics.Recall(task='multiclass', num_classes=num_classes+1, average=None, ignore_index=0).to(device)
            return(metric2(probs, targets)[1:])
        else:
            metric2 = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average=None, ignore_index=-1).to(device)
            return (metric2(probs, targets))
        