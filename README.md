# CAM
* CAM(Class Actication Maps) 논문 리뷰 및 구현
* 2023년 한양대학교 인공지능2 기말 과제
* 작성자 : 산업융합학부 정보융합전공 2020000055 유지수

🎓 본 논문 리뷰 및 구현은 기말 과제를 위해 작성되었습니다.
* CAM과 XAI를 처음 접하는 딥러닝 초보자를 위한 문서이며, CAM(Class Activation Maps)의 핵심 개념을 정리하고, CNN을 이용한 간단한 코드 구현을 통해 XAI의 대표적인 방법인 CAM은 어떻게 해석(시각화)이 가능한지에 대한 이해를 돕고자 작성되었습니다.

## 📘 CAM 논문 소개
* 세계최고권위 AI학회 CVPR에서 2016년에 발표된 “Learning Deep Features for Discriminative Localization” 논문입니다.
* MIT의 컵퓨터 과학 및 인공지능 연구소에서 발표되었습니다.
* CAM이라는 간단한 아이디어로 이미지 분류 모델 CNN(Convolution Neural Network)을 시각화하여 해석 가능한 방법을 제시하고 있습니다.

### 논문 핵심 요약
*	Global Average Pooling(GAP)를 적용하여 해석(시각화)가 가능한 구조를 제시합니다.
*	Feature Map에서 객체의 위치를 추출하는 방법인 Class Activation Mapping(CAM)을 제시합니다.
*	다양한 실험을 통해 논문에서 주장하는 구조와 객체 추출 방법이 객체 인식에 좋은 성능을 갖고 있음을 증명합니다.

### 문제점 및 해결책
#### 문제점
1. __Deep Learning = Black Box__ : 2016년도이지만 Deep Learning은 Black box모델이기 때문에 해석할 수 없다는 공통적인 한계가 존재했습니다.
2. 보통 CNN의 구조 = Input → Convolution Layers → __Fully connected Layers__ : CNN의 마지막 Layer를 FC-Layer로 Flatten하는 과정에서 Convolution이 가지고 있던 각 픽셀들의 위치 정보를 잃게 되므로 Classifying 정확도가 아무리 뛰어날 지라도 CNN이 특정 이미지의 어떤 Feature를 보고 특정 Class를 판별했는지 알 수 없습니다.

#### 해결책
1. __CAM 방법 활용__ : 위와 같은 문제로부터 연구되기 시작한 분야가 XAI(eXplainable AI)이며, XAI의 대표적인 방법인 CAM은  FC-Layer의 구조를 살짝 바꿔 위치정보를 손실하지 않도록 합니다.
2. CAM 방법 구조 = Input → Convolution Layers → __Global Average Pooling__ : 마지막 Convolution을 FC-Layer로 바꾸는 대신 GAP(Global Average Pooling)을 적용하여 별다른 추가의 지도학습 없이 CNN이 특정 위치들을 구별할 수 있도록 하고, CAM을 통해 특정 Class 이미지의 Heat Map을 생성하여 CNN이 어떻게 이 이미지를 특정 Class로 분류했는지를 이해할 수 있게 되어 Explainable한 결과를 낼 수 있도록 합니다.

### XAI : eXplainable AI, 설명가능 인공지능
* __뇌의 신경세포를 모방한 Neural Network__ : 신경망은 서로 복잡하게 연결된 수백만개 이상의 parameter(매개변수)가 비선형으로 상호작용하는 구조입니다. 사람이 그 많은 parameter를 직접 계산하고 의미를 파악하기는 불가능하고, back propagation(오류역전파법) 덕에 간신히 parameter update만 가능합니다.
* __Black Box Mode__ : 복잡한 구조 덕에 성능은 기존 기계학습보다도 월등히 높아지고 어떤 분야에서는 사람보다도 정확하지만 사람의 인지 영역을 넘어선 내부 구조 탓에 AI가 왜 그런 결과를 도출했는지는 개발자도 알 수 없게 되었습니다.
* __XAI (eXplainable AI)__ : XAI는 사람이 AI의 동작과 최종결과를 이해하고 올바르게 해석할 수 있고, 결과물이 생성되는 과정을 설명 가능하도록 해주는 기술을 의미합니다. 인공지능이 중요작업(mission critical)에 사용될 경우 인공지능의 설명성, 투명성 확보 기술, 기준 정립이 필수일 수밖에 없는 상황이 되었습니다.

### GAP(Global  Average Pooling) 구조
![GAP](https://github.com/KeepWater/CAM/assets/130841231/891f655d-cc1b-4711-a933-67bd0b51f82b)
* 모델의 구조는 크게 특징을 추출하는 Feature Extraction 단계와 추출된 특징을 이용하여 이미지를 분류하는 Classification 단계로 구분됩니다. Feature Extraction을 통해 생성된 Feature Map은 3차원 벡터(채널, 가로, 세로)이므로 이를 2차원 벡터(채널, 특징)로 변경하는 Flatten 단계를 수행하여 Classification 단계의 Input으로 활용합니다. Classification 단계는 여러 층의 Fully Connected Layer(FC Layer)로 구성되어 있으며 마지막 FC Layer에서 이미지를 분류합니다. Feature Extraction 단계에서 추출한 Feature Map은 여러 층의 FC Layer를 통과할 때 위치정보가 소실되므로 이 구조는 객체의 위치정보를 추출할 수 없습니다.
* 두번째 구조가 바로 GAP가 적용된 이미지 분류 모델 구조 예시이며, 카테고리 정보만을 학습하여 모델이 객체의 위치 추출 능력을 갖추기 위하여 본 논문에서는 Flatten 단계에서 Global Average Pooling(GAP) 방법을 사용해야 한다고 주장합니다.
* Global Average Pooling(GAP)은 각 Feature Map의 가로 세로 값을 모두 더하여 1개의 특징변수로 변환하는 것을 의미합니다. 예를 들어 위 그림에서는 총 4개의 Feature Map이 존재하므로 총 4개의 특징변수가 생성됩니다. 또한 Fully Connected Layer의 수를 줄이고 마지막 Classification Layer 하나만을 이용하여 모델을 구성합니다.

### CAM(Class Activation Mapping) 시각화 예시
* 앞서 언급했던 것처럼 Global Average Pooling을 사용함으로써 위치정보를 손실하지 않을 수 있도록 만들었는데, 덕분에 단 한 번의 forward-pass만을 통해 여러가지 Task를 수행하게 되었습니다. 예를 들어, Object Classification만을 위해 학습된 CNN 모델이 이미지를 classify할 수 있을 뿐만 아니라 localization도 수행할 수 있게 되었습니다. 즉, 각 이미지의 label만 주어진 상황에서 주어져 있지 않은 localization정보를 예측할 수 있게 됩니다.
![CAM](https://github.com/KeepWater/CAM/assets/130841231/d9525330-8896-4109-b454-aeb1dd3202fc)
* 위 그림은 Global Average Pooling을 사용하여 CAM을 시각화한 것인데, 각 이미지들에 대해 classify하면서도 object들이 위치하는 영역도 찾아낼 수 있음을 볼 수 있습니다.

### CAM 구조
* CAM은 CNN이 input image에 대한 prediction을 만들어냈을 때, 해당 class로 판별하는데 중요하게 생각하는 영역을 표시하여 시각화 하는 알고리즘입니다. convolution layer에서의 CAM을 시각화 했기 때문에, 최종 CAM을 처음 input image와 같은 크기로 upsampling하면, input image내에서 class와 관련되어 있는 영역이 어디인지 확인할 수 있습니다.
![CAM2](https://github.com/KeepWater/CAM/assets/130841231/6c36cfb4-89d6-46e5-a6d4-9f2b39704c7d)
* 위의 그림은 input image가 "Australian terrier"의 class로 구분되는데 영향을 미치는 영역을 하이라이트한 CAM 결과를 볼 수 있고, 강아지가 위치한 영역의 localization도 함께 수행한 것을 알 수 있습니다. Australian terrier를 분류하므로 사람의 얼굴에 집중한 첫 번째 activation map의 W1은 낮은 값일 것이고, 개의 특징에 주목한 두 번째, n 번째 activation map에 연결된 W2, Wn은 높은 값을 가질 것으로 유추할 수 있습니다.

### CAM 계산 과정
단계별 Calss Activation Mapping 계산 과정은 다음과 같습니다.
![CAM3](https://github.com/KeepWater/CAM/assets/130841231/92618b36-2b9a-4c57-a478-a28262776301)
1. 마지막 Convolution layer의 feature map을 fk(x, y)라고 하면 각각의 unit k에 대해 GAP을 수행해서 k개의 값을 출력합니다.(GAP의 결과 Fk)
2. 각각의 Fk에 대해서 class c에 대한 가중치의 weighted sum을 계산하여 Sc를 출력합니다.(softmax의 input으로 사용)
3. Softmax 연산을 거치면 각 class c에 대한 결과가 출력됩니다.(bias는 0으로 설정)
4. Class c에 대한 CAM을 Mc 라고 정의하고 Sc의 수식을 변형하여 구할 수 있는 형태로 사용합니다.

## 💻 CAM 구현
* 논문에서 제안한 __GAP__ 가 반영된 이미지 분류기 __ResNet__ 을 이용하여 실험합니다.
* Pytorch 기본 라이브러리에서 ImageNet을 이용하여 Pre-trained ResNet을 제공하고 있어 추가학습 없이 바로 __CAM__ 을 적용하여 시각화 하는 작업을 수행합니다.
* 코드는 Python 언어로 작성되었으며, Google Colab에서 진행했습니다.
* ~~저는 올해 정말 재밌었던 제주도 여행을 회상하며 제주도 바닷가에서 만난 말과 게의 사진으로 테스트를 해보았습니다.
CAM에 흥미를 갖고 계시다면 첨부파일을 다운받아 Google Drive에 업로드 한 뒤 본인이 원하는 사진 직접 테스트 해보는 것을 추천드립니다 :)~~
![crab](https://github.com/KeepWater/CAM/assets/130841231/8cb14555-f81e-410d-a760-c46d558b9de6)

### 라이브러리 및 모듈 임포트
```python
import torch
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import urllib.request
import ast
import numpy as np
import cv2
```

### 이미지 경로 및 클래스 정보 설정
```python
img_path = #본인이 테스트하고자 하는 이미지
classes_url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
with urllib.request.urlopen(classes_url) as handler:
    data = handler.read().decode()
    classes = ast.literal_eval(data)
```

### ResNet 모델 및 이미지 전처리 설정
```python
model_ft = models.resnet18(pretrained=True)
model_ft.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize])
```

### 이미지 전처리 및 모델 추론
```python
raw_img = Image.open(img_path)
img_input = preprocess(raw_img)
output = model_ft(img_input.unsqueeze(0))
softmaxValue = F.softmax(output)
class_id=int(softmaxValue.argmax().numpy())
```

### ResNet 구조 및 활성화 맵 관련 함수 정의
```python
def get_activation_info(self, x):
feature_maps = get_activation_info(model_ft, img_input.unsqueeze(0)).squeeze().detach().numpy()
activation_weights = list(model_ft.parameters())[-2].data.numpy()
```

### CAM, HeatMap 생성 및 시각화 함수 정의
```python
def show_CAM(numpy_img, feature_maps, activation_weights, classes, class_id):
show_CAM(numpy_img, feature_maps, activation_weights, classes, class_id)
```

### 🦀 결과
![image](https://github.com/KeepWater/CAM/assets/130841231/30848a95-b4e9-4471-bd24-b2d9cc0b246b)
* ResNet을 이용한 모델 및 클래스 추출 결과, "rock crab, Cancer irroratus"으로 잘 분류되었습니다.
* 클래스에 대한 CAM을 추출하고 HeatMap으로 시각화한 결과, 아래와 같이 바위게 "rock crab, Cancer irroratus" 클래스로 판별하는데 중요하게 생각하는 영역을 잘 표시함(객체의 좌표를 잘 추출함)을 확인할 수 있습니다.

## 📝 결론 및 느낀점
* 간단한 구조 변경만으로 다양한 Task(Classification, Localization)를 수행할 수 있는 방법을 제시한 점이 상당히 놀랍고 신기했습니다.
* CNN의 결과를 설명할 수 있는 CAM 방법을 구현해보며, Explainable model 뿐 아니라 직접 학습을 하지 않아도 localization에서도 좋은 성능을 이끌어내 수 있다는 점이 흥미로웠습니다.
* 부가적으로 CNN(Convolution Neural Network)의 작동 방식을 직관적으로 이해할 수 있었고, 본 논문에 나온 표현을 빌리면 __CNN의 영혼을 잠시 보는 접근 방법__ 을 제시한 것처럼 CNN이 우리가 생각한 대로 잘 작동하고 있다는 것을 알 수 있었습니다.
