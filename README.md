# PCCdata
##### 실행환경 COLAB
## Super Resolution업무관련 학부연구생 업무

### Point Cloud 일명 PCC 3D 자료를 2D로 Projection 한 결과를 yuv 파일로 변형후 super resolution 모델로 PSNR 측정한다


1. YUV 파일을 ffmpeg application 을 통해 png 파일로 변환한다.
2. 변환된 png 파일 을 super resolution 모델에 적용한다.( url 에서 불러오는것과 local 에 저장해서 하는것은 directory 설정을 잘해주어야함 하위폴더를 잘맞춰주어야한다)
3. 변환된 파일을 excel로 저장한다.
4. 제출


### 아주 간단한 CNN 모델을 적용하였지만 추후에는  U NET ,RES NET 을 적용할것임.



## 2021 7월 29일자 

### Predicting the perceptual Quality of Point cloud A 3D to 2D projection based exploration 논문리뷰
##### 저자 Qi Yang, Hao Chen, Zhan Ma, Senior Member, IEEE, Yiling Xu, Member, IEEE, Rongjun Tang, and Jun Sun


이 논문에서는 어떻게 Point Cloud 의 quality 를 높일 것인가에 대한 고찰을 담겨있다.

배경: 기존 Point Cloud 는 point to point (p2point) , point to plane (p2plane),point to mesh(p2mesh)
는 euclidean distance 를 metric 으로 사용하게되는데 이는 오류가많다. 단순 거리의로 계산하므로 실제 사람이 보는 지표와는 거리가 멀다.

따라서 방법이 필요하다.

MPEG,JPEG 연구회에서 많은 연구를 하고있지만 뚜렷한 성과가없다.

PCC 의 distortion 의 원인은 geometry, photometric attributes 에있다.

### 이들의 방식은 3D 를 육면체 안에 담는다.

그리고 3D 를 각 육면체에 Projection 시킨다(정사영시킨다) 
그러면 6개의 projection 된 결과가 나오게되는데 이것은 2D 이므로 계산하기 수월해진다

우리는 대부분 2D로 다루는것을 좋아하기때문이다,


### Data processing

X(m,i)는 i번째 대상의 m번째 point cloud이다.

<img width="463" alt="스크린샷 2021-07-30 오전 2 22 39" src="https://user-images.githubusercontent.com/76778082/127536888-2664bc16-0062-49c5-a4fa-1b44b335890f.png">

### 먼저 Z 정규화를 해준다. 


<img width="463" alt="스크린샷 2021-07-30 오전 2 24 13" src="https://user-images.githubusercontent.com/76778082/127537082-d809877b-124b-4392-bdd4-d37a98fb5731.png">

### 그다음 rescaling 을 해준다 , 해당 식을 보면 (max-min)*Z에대한식 +min 인데 이는 잘생각해보면 min과 max 사이에있게 만들어주기위한 식이다.
간단하게 예를들어 보자 (max-min)을 하나의 방향벡터라고 보고 Z에대한식 ( max-min 정규화 이므로) 0과1사이의 값을 가지게된다 
결국 min+ (max-min)*p (p는 0이상 1이하의 값) 이될텐데 이는 결국 min 과 max 사이에있다.

해당 논문에서는 min을 1 max 를 10으로 사용하였다.


## Why Do we have to conver RGB to YUV ???

이에대한 답변은 간단하다.

RGB의 채널은 강한 correlation 을 가지고있는데 이는 서로 상관관계가 높다는 뜻이다. 
이를 Gausian image matrix 를 곱해 YUV 채널로 만들어주자 ( Decompose 가 된다)



<img width="463" alt="스크린샷 2021-07-30 오전 2 29 48" src="https://user-images.githubusercontent.com/76778082/127537807-e666d808-35a5-4b39-82e5-767a33d35065.png">

### Feature extraction


<img width="463" alt="스크린샷 2021-07-30 오전 2 30 22" src="https://user-images.githubusercontent.com/76778082/127537897-60502a0e-c343-4203-bf88-cb0050e9f5f4.png">

### local 과 global 의 곱으로 표현된다.


<img width="463" alt="스크린샷 2021-07-30 오전 2 31 14" src="https://user-images.githubusercontent.com/76778082/127538003-8d52a2d3-ba08-4c6f-b5c0-accc90ba5d00.png">


### Sobel Detector 를 활용한다.

Sobel Detector 란?

음.. 여기서 글로표현하기란 애매하니까 발표자료의 스크린샷으로 대체함.

<img width="463" src=https://user-images.githubusercontent.com/76778082/127538728-9528b9cf-f141-4eb6-9fec-e1b70f593157.jpeg>

글씨는 양해부탁드립니다..

<img width="463" alt="스크린샷 2021-07-30 오전 2 37 14" src="https://user-images.githubusercontent.com/76778082/127538854-a6b1fbf5-7c28-4b4f-a37c-3a0eb354797b.png">


이식에대한 의미는 정확히 모르겠으나 확률*log(확률) 이므로  xlogx 로 바꿔생각해볼수있다. 
즉 -log( 적분 xlogx dx) 정도인데  log 함수 특성상 0에 가까이가면 빠르게줄어든다. 그러한특성을 반영한게 아닐까 싶다.
미분해서 그래프 그려봐도 1/e 에서 global minimum 을 가지는거밖에 체크할수가없다. 무슨의미인지모르겠다.


<img width="463" src=https://user-images.githubusercontent.com/76778082/127540428-d11808fd-f4f1-46f3-aec3-1158dea28350.png>


### 다음 스크린샷은 KL 쿨백 라이블러 divergence 와 옌센에 대한 식인데
이것은 간단하게 설명하자면 KL 은 두확률간의 분포유사도 측정인데 metric 으로 사용할수가없다. 이유는 비대칭적이라서 그렇다

옌센은 이를 보완한 대칭적인함수이다, 따라서 metric 으로 사용할수있다 .이 값의 root를 씌워준 값으로 사용하게된다.

이러한 지표들을 사용하여 평가한다.

일단 리뷰는 여기까지만 하겠습니다. 후에 피어슨 상관계수도나오는데 이것은 별로 어렵지않다








