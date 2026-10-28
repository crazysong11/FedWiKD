"""Teacher and student CNNs and their analytical operation counts."""

import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(width, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(width, momentum=0.9)
        self.pool = nn.MaxPool2d(2)
        self.feature_dim = width * 4 * 4
        self.fc_hidden = 16 * width
        self.fc1 = nn.Linear(self.feature_dim, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 10)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, self.feature_dim)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        features = self.extract_features(x)
        features = self.dropout(features)
        return self.fc2(features)


class StudentCNN(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(width, momentum=0.9)
        self.pool = nn.MaxPool2d(2)
        self.feature_dim = width * 8 * 8
        self.fc_hidden = 4 * width
        self.fc1 = nn.Linear(self.feature_dim, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 10)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, self.feature_dim)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        features = self.extract_features(x)
        features = self.dropout(features)
        return self.fc2(features)


def get_model(arch_type: str, width: int):
    if arch_type == "teacher":
        return TeacherCNN(width=width)
    elif arch_type == "student":
        return StudentCNN(width=width)
    else:
        raise ValueError(f"Unknown arch_type: {arch_type!r}. Supported: 'teacher', 'student'")


def get_teacher_model(width: int):
    return get_model("teacher", width)


def get_student_model(width: int):
    return get_model("student", width)


def get_full_model(width: int):
    return get_model("teacher", width)


def model_param_count_formula(arch: str, width: int) -> int:
    if arch == "teacher":
        return 274 * width**2 + 212 * width + 10
    if arch == "student":
        return 265 * width**2 + 77 * width + 10
    raise ValueError(f"Unknown arch={arch!r}")


def model_forward_flops_formula(arch: str, width: int) -> int:
    if arch == "teacher":
        return 6272 * width**2 + 55616 * width
    if arch == "student":
        return 5120 * width**2 + 55376 * width
    raise ValueError(f"Unknown arch={arch!r}")


def model_feature_dim_formula(arch: str, width: int) -> int:
    if arch == "teacher":
        return 16 * width
    if arch == "student":
        return 4 * width
    raise ValueError(f"Unknown arch={arch!r}")
