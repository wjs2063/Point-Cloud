# PCCdata
##### 실행환경 COLAB
## Point cloud Super Resolution 학부연구생 업무
주제: Point Cloud Perceptual quality 
3D 입체 데이터를 2D로 변환시킨자료를 어떻게 평가할것인가?와 품질을 어떻게 높일수있는가? 이다

맡은 업무: 기본 keras tutorial model로 Projection 한 결과의 PSNR 측정, 그리고 UNET 의 구조를 적용한후 PSNR 측정후 정리 + 논문리뷰 이다.


## 업무관련 진행사항 요약 및 설명

### Point Cloud 일명 PCC 3D 자료를 2D로 Projection 한 결과를 yuv 파일로 변형후 super resolution 모델로 PSNR 측정한다


1. YUV 파일을 ffmpeg application 을 통해 png 파일로 변환한다. 
2. 변환된 png 파일 을 super resolution 모델에 적용한다.
3. 변환된 파일을 excel로 저장한다.
4. 제출

### 아주 간단한 CNN 모델을 적용하였지만 추후에는  U NET ,RES NET 을 적용예정

## 2020-04 Subpixel 을 활용한 superresolution 코드 설명
```
crop_size=300
upscale_factor=3
input_size=crop_size//upscale_factor
batch_size=8

train_ds=image_dataset_from_directory(
    base_dir,
    batch_size=batch_size,
    image_size=(crop_size,crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
    color_mode='rgb',


)
valid_ds=image_dataset_from_directory(
    base_dir,
    batch_size=batch_size,
    image_size=(crop_size,crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
    color_mode='rgb',
    
      
)
   
def scaling(input_image):
    input_image=input_image/255.0
    return input_image
train_ds=train_ds.map(scaling)
valid_ds=train_ds.map(scaling)
```

해당 tutorial 모델에서는 url 에서 불러오는 방식을 선택하게된다. 하지만 local directory 에서 불러오게된다면?
오류가뜬다.
이유는 url 에서불러오게되면 자동으로 디렉토리가만들어지는데 상위 디렉토리 -> 하위디렉토리1->하위디렉토리1의 하위디렉토리에 image 들이 저장되어있어야한다. 이 구조를 꼭 지켜주어야함.
이 구조만 지켜준다면 이후 코드실행은 문제없이 구동될것이다



## Build model
```
def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(128, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
```
## Compile and train

```
epochs = 100

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)


model.load_weights(checkpoint_filepath)
```



## 2021 7월 29일자 

### Predicting the perceptual Quality of Point cloud A 3D to 2D projection based exploration 논문리뷰
##### 저자 Qi Yang, Hao Chen, Zhan Ma, Senior Member, IEEE, Yiling Xu, Member, IEEE, Rongjun Tang, and Jun Sun


이 논문에서는 어떻게 Point Cloud 의 quality 를 높일 것인가에 대한 고찰을 담겨있다.

배경: 기존 Point Cloud 는 point to point (p2point) , point to plane (p2plane),point to mesh(p2mesh)
는 euclidean distance 를 metric 으로 사용하게되는데 이는 오류가많다. 단순 거리의로 계산하므로 실제 사람이 보는 지표와는 거리가 멀다.

MPEG,JPEG 연구회에서 많은 연구를 하고있지만 뚜렷한 성과가없다.

따라서 새로운 방법이 필요하다.

PCC 의 distortion 의 원인은 geometry, photometric attributes 에있다.

### 이들의 방식은 3D 를 육면체 안에 담는다.

그리고 3D 를 6개의면에  Projection 시킨다(정사영시킨다) 
그러면 6개의 projection 된 결과가 나오게되는데 이것은 2D 이므로 계산하기 수월해진다

우리는 대부분 2D로 다루는것을 좋아하기때문이다,( 2D로 변환하면 적용하기 편해짐)


### Data processing

X(m,i)는 i번째 대상의 m번째 point cloud이다.

<img width="463" alt="스크린샷 2021-07-30 오전 2 22 39" src="https://user-images.githubusercontent.com/76778082/127536888-2664bc16-0062-49c5-a4fa-1b44b335890f.png">

### 먼저 Z 정규화를 해준다. 


<img width="463" alt="스크린샷 2021-07-30 오전 2 24 13" src="https://user-images.githubusercontent.com/76778082/127537082-d809877b-124b-4392-bdd4-d37a98fb5731.png">

### 그다음 rescaling 을 해준다 , 해당 식을 보면 (max-min)*Z에대한식 +min 인데 이는 잘생각해보면 min과 max 사이에있게 만들어주기위한 식이다.
간단하게 예를들어 보자 (max-min)을 하나의 방향벡터라고 보고 Z에대한식 ( max-min 정규화 이므로)을 크기를 담당하는 부분으로보자(0과1사이로 줄였다가 늘였다가)
결국 min+ (max-min)*p (p는 0이상 1이하의 값) 이될텐데 이는 결국 min 과 max 사이에있다.

해당 논문에서는 min을 1 max 를 10으로 사용하였다.


## Why Do we have to convert RGB to YUV ???

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



<img width="463" alt="스크린샷 2021-07-30 오전 2 37 14" src="https://user-images.githubusercontent.com/76778082/127538854-a6b1fbf5-7c28-4b4f-a37c-3a0eb354797b.png">


I=1/P  정보량과 확률의 관계이다.  여기서 -x*log(p(x)) 는 x *log(1/p) 로 볼수있는데 이것은 결국  엔트로피 값을 나타내게 된다
해당 확률변수가 지니는 평균적인 불확실성의 정도이다.


<img width="463" src=https://user-images.githubusercontent.com/76778082/127540428-d11808fd-f4f1-46f3-aec3-1158dea28350.png>


### 다음 스크린샷은 KL 쿨백 라이블러 divergence 와 옌센에 대한 식인데
이것은 간단하게 설명하자면 KL 은 두확률간의 분포유사도 측정인데 metric 으로 사용할수가없다. 이유는 비대칭적이라서 그렇다

옌센은 이를 보완한 대칭적인함수이다, 따라서 metric 으로 사용할수있다 .이 값의 root를 씌워준 값으로 사용하게된다.

이러한 지표들을 사용하여 평가한다.

일단 리뷰는 여기까지만 하겠습니다. 후에 피어슨 상관계수도나오는데 이것은 별로 어렵지않다








