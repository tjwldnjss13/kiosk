import torch.nn as nn

from models.fortest.attention_modules_modified import CBAM
from models.fortest.mobilenetv2_cbam import MobilentV2CBAM


class AGNetCBAMSuperSub(nn.Module):
    def __init__(self):
        super().__init__()
        # 15~19, 20~24, 25~29, 30~34, 35~39, 40~44, 45~49, 50~54, 55~59+
        self.layers = MobilentV2CBAM()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_age_super = nn.Linear(1280, 9)
        self.out_age_subs = nn.Sequential(*[nn.Linear(1, 5) for _ in range(9)])
        # self.out_age_sub1 = nn.Linear(1, 5)
        # self.out_age_sub2 = nn.Linear(1, 10)
        # self.out_age_sub3 = nn.Linear(1, 10)
        # self.out_age_sub4 = nn.Linear(1, 10)
        # self.out_age_sub5 = nn.Linear(1, 10)
        # self.out_age_sub6 = nn.Linear(1, 1)
        self.out_gender = nn.Linear(1280, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.attention = CBAM(1280, 1/16)
        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)

        age_super = self.out_age_super(x)
        gender = self.out_gender(x)

        # age_subs = []
        # for i, out_age_sub in enumerate(self.out_age_subs):
        #     age_subs.append(out_age_sub(age_super[..., i].unsqueeze(-1)))

        age_subs = [self.out_age_subs[i](age_super[..., i].unsqueeze(-1)) for i in range(age_super.shape[-1])]

        # age_sub1 = self.out_age_sub1(age_super[..., 0].unsqueeze(-1))
        # age_sub2 = self.out_age_sub2(age_super[..., 1].unsqueeze(-1))
        # age_sub3 = self.out_age_sub3(age_super[..., 2].unsqueeze(-1))
        # age_sub4 = self.out_age_sub4(age_super[..., 3].unsqueeze(-1))
        # age_sub5 = self.out_age_sub5(age_super[..., 4].unsqueeze(-1))
        # age_sub6 = self.out_age_sub6(age_super[..., 5].unsqueeze(-1))

        age_super = self.softmax(age_super)
        age_subs = [self.softmax(age_sub) for age_sub in age_subs]
        # age_sub1 = self.softmax(age_sub1)
        # age_sub2 = self.softmax(age_sub2)
        # age_sub3 = self.softmax(age_sub3)
        # age_sub4 = self.softmax(age_sub4)
        # age_sub5 = self.softmax(age_sub5)
        # age_sub6 = self.softmax(age_sub6)
        gender = self.softmax(gender)

        # return age_super, [age_sub1, age_sub2, age_sub3, age_sub4, age_sub5, age_sub6], gender
        return age_super, age_subs, gender


if __name__ == '__main__':
    from torchsummary import summary
    model = AGNetCBAMSuperSub().cuda()
    summary(model, (3, 224, 224))
