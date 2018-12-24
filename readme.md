# File describtion
![image](https://user-images.githubusercontent.com/33209778/50393910-ab323e00-079c-11e9-9119-cd432e447a18.png)  

*.ttf : font file
draw_CNN.ipynb : draw figure of CNN  
font_conv.py : train network  
font_conv_test.py : test  
font_convnet_real.ipynb : test in jupyter notebook(use finally)  
hangul_unicode.ipynb : font file to font image  
trainset.txt, testset.txt : train, test image dir  
testset2.txt : input image dir for prediciton  
  
# Font Conversion Convolutional Network
## [1] Dataset
### a. 64 * 64 미생체 폰트 이미지 (11172자 지원): 입력 데이터로 사용
### b. 64 * 64 연성체 폰트 이미지 (2436자 지원): target 데이터로 사용
이미지 생성시 지원하지않는 글자는 흰색 배경만 출력함.  

|          | all supporting  (# = 2436) | only misaeng supporting (# = 11172 - 2436) |
|:--------:|:------------------------:|:-----------------------------------------:|
|  misaeng |    ![image](https://user-images.githubusercontent.com/33209778/50393125-5dff9d80-0797-11e9-8d30-4e5be80a5bb8.png) ![image](https://user-images.githubusercontent.com/33209778/50393131-622bbb00-0797-11e9-8907-b85afeeab67b.png) ![image](https://user-images.githubusercontent.com/33209778/50393132-648e1500-0797-11e9-891b-563999e814fa.png)                    |           ![image](https://user-images.githubusercontent.com/33209778/50393173-a61ec000-0797-11e9-8c7f-929d1fe8e799.png) ![image](https://user-images.githubusercontent.com/33209778/50393175-a8811a00-0797-11e9-8e56-359f91ed54bf.png) ![image](https://user-images.githubusercontent.com/33209778/50393176-aa4add80-0797-11e9-864e-4642d171a243.png) ![image](https://user-images.githubusercontent.com/33209778/50393178-ad45ce00-0797-11e9-8b90-243857764371.png)      |
| yeonsung |            ![image](https://user-images.githubusercontent.com/33209778/50393155-8c7d7880-0797-11e9-8fb5-ec54f3983945.png) ![image](https://user-images.githubusercontent.com/33209778/50393159-91422c80-0797-11e9-9573-1d9dd4c21f11.png) ![image](https://user-images.githubusercontent.com/33209778/50393163-956e4a00-0797-11e9-82b1-9146b9ecbd16.png)              |                     -                     |

-> 연성체 지원 글자에 맞추어 train, test (4:1) 진행.  
-> 연성체 지원 글자가 아닌 글자를 미생체로 입력하여 출력을 정성적 확인.  

## [2] Network
### a. layer5
![image](https://user-images.githubusercontent.com/33209778/50393348-06623180-0799-11e9-9735-8917a2d43821.png)

### b. layer10
![image](https://user-images.githubusercontent.com/33209778/50393357-10843000-0799-11e9-9c99-2cf1ad20ce12.png)
![image](https://user-images.githubusercontent.com/33209778/50393358-137f2080-0799-11e9-9bc4-af2e0b4de574.png)

### c. loss function, optimizer
Loss function: input font와 target font사이 2D L1 Loss 사용  
Optimizer: Adam 알고리즘 사용  

## [3] Result
모든 결과에서 학습 epoch이 더 적음에도 비교적 layer가 깊을 때 정성적으로 더 잘 conversion되는 것을 확인할 수 있습니다.  
Layer10의 학습을 더 오래 시킨다면 더 적어질 가능성이 보입니다.  

### a. Train, Test(all supporting)
#### 1) layer5
300 epoch training  
minimum training L1 loss = 0.13  
testing L1 loss = 0.146  
  
| content | font image |
|---------|------------|
| Input   |  ![image](https://user-images.githubusercontent.com/33209778/50393505-01ea4880-079a-11e9-848a-97b06b5648bd.png)  |
| Output  |  ![image](https://user-images.githubusercontent.com/33209778/50393541-470e7a80-079a-11e9-83b2-89c7d2c9ad07.png)  |
| Target  |  ![image](https://user-images.githubusercontent.com/33209778/50393530-39f18b80-079a-11e9-94b0-4e5f00b2d3ba.png)  |
  
#### 2) layer10
114 epoch training  
minimum training L1 loss = 0.098  
testing L1 loss = 0.130  
  
| content | font image |
|---------|------------|
| Input   |  ![image](https://user-images.githubusercontent.com/33209778/50393505-01ea4880-079a-11e9-848a-97b06b5648bd.png)  |
| Output  |  ![image](https://user-images.githubusercontent.com/33209778/50393564-6b6a5700-079a-11e9-9766-2c63d48c5860.png)  |
| Target  |  ![image](https://user-images.githubusercontent.com/33209778/50393530-39f18b80-079a-11e9-94b0-4e5f00b2d3ba.png)  |
  
  
#### 3) layer10
2000 epoch training  
  
| content | font image |
|---------|------------|
| Input   |  ![image](https://user-images.githubusercontent.com/33209778/50393505-01ea4880-079a-11e9-848a-97b06b5648bd.png)       |
| Output  |    ![image](https://user-images.githubusercontent.com/33209778/50393694-488c7280-079b-11e9-8671-b9749c486729.png)    |
| Target  |        ![image](https://user-images.githubusercontent.com/33209778/50393530-39f18b80-079a-11e9-94b0-4e5f00b2d3ba.png)  |
  
### b. prediction (only misaeng supporting)
  
| content | font image |
|---------|------------|
|  Input  |  ![image](https://user-images.githubusercontent.com/33209778/50393763-c18bca00-079b-11e9-9d34-779d2c54c69f.png)  |
|  Layer5  |  ![image](https://user-images.githubusercontent.com/33209778/50393783-f435c280-079b-11e9-9bf8-58a54252a99f.png)  |
|  Layer10, epoch100  |  ![image](https://user-images.githubusercontent.com/33209778/50393801-16c7db80-079c-11e9-8e40-95f59da9d43c.png)  |
|  Layer10, epoch2000  |  ![image](https://user-images.githubusercontent.com/33209778/50393805-1deee980-079c-11e9-85a4-56385aa2f4d4.png)  |
