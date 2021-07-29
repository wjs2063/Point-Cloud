# PCCdata

# Super Resolution업무관련 학부연구생 업무

## Point Cloud 일명 PCC 3D 자료를 2D로 Projection 한 결과를 yuv 파일로 변형후 super resolution 모델로 PSNR 측정한다


1. YUV 파일을 ffmpeg application 을 통해 png 파일로 변환한다.
2. 변환된 png 파일 을 super resolution 모델에 적용한다.( url 에서 불러오는것과 local 에 저장해서 하는것은 directory 설정을 잘해주어야함 하위폴더를 잘맞춰주어야한다)
3. 변환된 파일을 excel로 저장한다.
4. 제출


## 아주 간단한 CNN 모델을 적용하였지만 추후에는 조금더 향상된 U NET ,RES NET 을 적용하는건어떨까?
