# README
## 主題
本次實作HDR演算法，通過將多張不同曝光時間的LDR影像合成為品質較佳的HDR影像，並以tone mapping技術將其再轉換為更加清晰的LDR圖片
## 實作細節
首先會將圖片進行一定程度的alignment，藉此減少拍攝過程中晃動造成的誤差，之後經由Paul Debevec’s method將其轉換為HDR影像，取得HDR影像後，便可經由Photographic global tone mapping/Photographic local tone mapping/Adaptive logarithmic mapping產生出品質更佳的LDR影像
## result
![](./figure/result1.png)
## 執行
首先需要建立一個input file的list  
格式如下:  
./example1.file time1  
./example2.file time2  
...  
其中第一項為照片的路徑(絕對路徑或相對於hw1.py的路徑)  
第二項為曝光時間的倒數  
如下圖:  
![](https://i.imgur.com/ShdxOpr.png)  
接下來執行  
```shell
python3 HDR.py input [-o OUTPUT] [-c CURVE]
```
input: input file list(mentioned above)  
output: out put directory(default is ./result)  
curve: show curve(default is False)  
執行時間大約數分鐘至數十分鐘不等  
## 輸出

輸出為:  
./result/HDR_img.hdr 為hdr輸出檔  
./result/Ldr_global.png 為使用photographic global tone mapping的LDR image  
./result/Ldr_local.png 為使用photographic local tone mapping的LDR image  
./result/Ldr_log.png 為使用adaptive logarithmic mapping的LDR image  
./result/curve/Curve_0.png  
./result/curve/Curve_1.png  
./result/curve/Curve_2.png 為BGR的response curve結果  
## package
本次使用的python額外套件  
cv2  
numpy  
matplotlib  

