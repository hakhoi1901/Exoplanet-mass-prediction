# Ví dụ của cô

## 1. Giới thiệu

**Mục tiêu:**

- Minh họa quy trình xử lý và phân tích một bộ dữ liệu thực tế
- Xây dựng và đánh giá một mô hình hồi quy tuyến tính

**Bài toán:** Yếu tố nào đang ảnh hưởng đến doanh thu phim? (Ngân sách sản xuất / Thể loại phim / Đánh giá của khán giả / Số lượng rạp chiếu /…? ) Liệu có thể xây dựng một mô hình toán học để dự đoán doanh thu phim dựa trên các yếu tố này không ?

→ Đây là một ví dụ điển hình của bài toán ***data fiting,*** trong đó mục tiêu là tìm ra một hàm số phù hợp nhất để mô tả mối quan hệ giữa biến đầu ra (doanh thu) và các biến đầu vào (các đặc trưng của phim)

**Quy ước:** mức ý nghĩa áp dụng cho các kiểm định $a = 0.05$

*Tài liệu này dành riêng cho mục đích học thuật và cách xử lý này chưa phải là tốt nhất*

## 2. Hiểu dữ liệu (Data Understanding)

Mục đích: làm quen với bộ dữ liệu, đánh giá chất lượng dữ liệu ban đầu và khám phá các đặc điểm quan trọng trước khi tiến hành tiền xử lý dữ liệu và xây dựng mô hình

### 2.1 Mô tả dữ liệu (Data Description)

- Nguồn dữ liệu
    
    Trong Case Study này, chúng ta sử dụng bộ dữ liệu **CSM (Conventional and Social Media Movies Dataset 2014 and 2015)** lấy từ nguồn UCI Machine Learning Repository:
    [https://archive.ics.uci.edu/dataset/424/csm+conventional+and+social+media+movies+dataset+2014+and+2015](https://archive.ics.uci.edu/dataset/424/csm+conventional+and+social+media+movies+dataset+2014+and+2015)
    
    *CSM Dataset* tổng hợp thông tin về các bộ phim được phát hành trong hai năm 2014 và 2015, kết hợp giữa các đặc trưng truyền thống của ngành điện ảnh và các chỉ số liên quan đến mạng xã hội.
    
    ### Nạp dữ liệu:
    
    Dưới đây là cấu trúc dữ liệu mẫu (tibble 6 × 14):
    
    | **Movie** | **Year** | **Ratings** | **Genre** | **Gross** | **Budget** | **Screens** | **Sequel** | **Sentiment** | **Views** | **Likes** |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 13 Si... | 2014 | 6.3 | 8 | 9.13e3 | 4e6 | 45 | 1 | 0 | 3.28e6 | 4632 |
    | 22 Ju... | 2014 | 7.1 | 1 | 1.92e8 | 5e7 | 3306 | 2 | 2 | 5.83e5 | 3465 |
    | 3 Day... | 2014 | 6.2 | 1 | 3.07e7 | 2.8e7 | 2872 | 1 | 0 | 3.05e5 | 328 |
    | 300: ... | 2014 | 6.3 | 1 | 1.06e8 | 1.10e8 | 3470 | 2 | 0 | 4.53e5 | 2429 |
    | A Hau... | 2014 | 4.7 | 8 | 1.73e7 | 3.5e6 | 2310 | 2 | 0 | 3.15e6 | 12163 |
    | A Lon... | 2014 | 4.6 | 3 | 2.90e4 | 5e5 | NA | 1 | 0 | 9.11e4 | 112 |
    
    **Ghi chú:** Ngoài các cột trên, bộ dữ liệu còn 3 biến khác: `Dislikes`, `Comments`, và `Aggregate Followers`.
    
- Mô tả dữ liệu
    
    Kiểm tra quy mô dữ liệu:
    
    ```
    ## [1] 231 14
    ```
    
    Bộ dữ liệu gồm **231 quan trắc** (bộ phim), **14 đặc trưng** (biến) mô tả đặc điểm của từng bộ phim:
    
    - **“Movie”**: tên phim
    - **“Year”**: năm phát hành
    - **“Ratings”**: điểm đánh giá trung bình của bộ phim
    - **“Genre”**: thể loại phim
    - **“Gross”**: tổng doanh thu phim
    - **“Budget”**: ngân sách sản xuất (tổng chi phí)
    - **“Screens”**: số lượng rạp chiếu
    - **“Sequel”**: phần phim
    - **“Sentiment”**: điểm cảm xúc của khán giả
    - **“Views”**: số lượt xem
    - **“Likes”**: số lượt thích
    - **“Dislikes”**: số lượt không thích
    - **“Comments”**: số bình luận
    - **“Aggregate Followers”**: số người theo dõi
    
    → Chọn biến mục tiêu (biến đầu ra/ biến đáp ứng) để tìm ra mô hình tốt nhất giải thích cho tổng doanh thu phim: **Gross** - tổng doanh thu phim.
    
    Đổi tên biến *Aggregate Followers* thành *AggregateFollowers* (cho dễ thao tác)
    

### 2.2 Khám phá dữ liệu (Explore Data)

- Thông tin cơ bản
    
    Kiểm tra sơ liệu về kiểu dữ liệu và loại dữ liệu:
    
    ```
    ## tibble [231 × 14] (S3: tbl_df/tbl/data.frame)
    ##  $ Movie             : chr [1:231] "13 Sins" "22 Jump Street" "3 Days to Kill" "300: Rise of an Empire" ...
    ##  $ Year              : num [1:231] 2014 2014 2014 2014 2014 ...
    ##  $ Ratings           : num [1:231] 6.3 7.1 6.2 6.3 4.7 4.6 6.1 7.1 6.5 6.1 ...
    ##  $ Genre             : num [1:231] 8 1 1 1 8 3 8 1 10 8 ...
    ##  $ Gross             : num [1:231] 9.13e+03 1.92e+08 3.07e+07 1.06e+08 1.73e+07 2.90e+04 4.26e+07 ...
    ##  $ Budget            : num [1:231] 4.00e+06 5.00e+07 2.80e+07 1.10e+08 3.50e+06 5.00e+05 4.00e+07 ...
    ##  $ Screens           : num [1:231] 45 3306 2872 3470 2310 ...
    ##  $ Sequel            : num [1:231] 1 2 1 2 2 1 1 1 1 1 ...
    ##  $ Sentiment         : num [1:231] 0 2 0 0 0 0 2 3 0 ...
    ##  $ Views             : num [1:231] 3280543 583289 304861 452917 3145573 ...
    ##  $ Likes             : num [1:231] 4632 3465 328 2429 12163 ...
    ##  $ Dislikes          : num [1:231] 425 61 34 132 610 7 419 197 419 532 ...
    ##  $ Comments          : num [1:231] 636 186 47 590 1082 ...
    ##  $ AggregateFollowers: num [1:231] 1120000 12350000 483000 568000 1923800 ...
    ```
    
    *Sinh viên cần đưa ra nhận định cụ thể về kiểu dữ liệu và loại dữ liệu của từng đặc trưng cụ thể, cùng với nhận xét của cá nhân mình để hiểu rõ về dữ liệu.*
    
    Sau khi kiểm tra sơ liệu về bộ dữ liệu, ta nhận thấy bộ dữ liệu có một số biến sau đây:
    
    - **“Year”**: gồm 2 năm phát hành của phim: 2014 và 2015, đang được lưu theo kiểu số.
    - **“Genre”**: gồm 11 thể loại phim, đang được lưu theo kiểu số.
    - **“Sequel”**: gồm 7 phần phim, đang được lưu theo kiểu số.
    
    Đây là các biến định tính, không phải biến định lượng.
    
    Các biến còn lại đều có kiểu dữ liệu phù hợp.
    
    Để đảm bảo tính chất phân loại của các biến này được đưa vào mô hình đúng ý nghĩa, giúp các thuật toán hồi quy xử lý dữ liệu chính xác hơn, ta tiến hành chuẩn hóa kiểu dữ liệu của chúng thành định tính:
    
    ```
    ## tibble [231 × 14] (S3: tbl_df/tbl/data.frame)
    ##  $ Movie             : chr [1:231] "13 Sins" "22 Jump Street" "3 Days to Kill" "300: Rise of an Empire" ...
    ##  $ Year              : Factor w/ 2 levels "2014","2015": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ Ratings           : num [1:231] 6.3 7.1 6.2 6.3 4.7 4.6 6.1 7.1 6.5 6.1 ...
    ##  $ Genre             : Factor w/ 11 levels "1","2","3","4",..: 7 1 1 1 7 3 7 1 9 7 ...
    ##  $ Gross             : num [1:231] 9.13e+03 1.92e+08 3.07e+07 1.06e+08 1.73e+07 2.90e+04 4.26e+07 ...
    ##  $ Budget            : num [1:231] 4.00e+06 5.00e+07 2.80e+07 1.10e+08 3.50e+06 5.00e+05 4.00e+07 ...
    ##  $ Screens           : num [1:231] 45 3306 2872 3470 2310 ...
    ##  $ Sequel            : Factor w/ 7 levels "1","2","3","4",..: 1 2 1 2 2 1 1 1 1 1 ...
    ##  $ Sentiment         : num [1:231] 0 2 0 0 0 0 2 3 0 ...
    ##  $ Views             : num [1:231] 3280543 583289 304861 452917 3145573 ...
    ##  $ Likes             : num [1:231] 4632 3465 328 2429 12163 ...
    ##  $ Dislikes          : num [1:231] 425 61 34 132 610 7 419 197 419 532 ...
    ##  $ Comments          : num [1:231] 636 186 47 590 1082 ...
    ##  $ AggregateFollowers: num [1:231] 1120000 12350000 483000 568000 1923800 ...
    ```
    
    ### Về giá trị dữ liệu:
    
    - Giá trị của các biến trong bộ dữ liệu có sự chênh lệch lớn giữa các biến.
    - Biến **“Sentiment”**: có 119 giá trị $\le 0$ và 112 giá trị $> 0$.
    - Biến **“Dislikes”** có 4 giá trị $= 0$, các giá trị còn lại đều $> 0$.
- Kiểm tra chất lượng dữ liệu
    
    Để đánh giá chất lượng dữ liệu, ta tiến hành kiểm tra xem dữ liệu có trùng lặp (duplicate), có khuyết (missing), có nhiễu (noise), có mâu thuẫn (conflict) hay không.
    
    ### Kiểm tra dữ liệu trùng lặp (duplicate):
    
    ```
    ## [1] 0
    ```
    
    Ta thấy bộ dữ liệu không có dữ liệu trùng lặp.
    
    ### Kiểm tra dữ liệu khuyết (missing):
    
    ```
    ##             Movie              Year           Ratings             Genre
    ##                 0                 0                 0                 0
    ##             Gross            Budget           Screens            Sequel
    ##                 0                 1                10                 0
    ##         Sentiment             Views             Likes          Dislikes
    ##                 0                 0                 0                 0
    ##          Comments AggregateFollowers
    ##                 0                35
    ```
    
    Ta thấy bộ dữ liệu có:
    
    - 1 giá trị khuyết của biến **“Budget”**
    - 10 giá trị khuyết của biến **“Screens”**
    - 35 giá trị khuyết của biến **“AggregateFollowers”**
    
    ### Kiểm tra dữ liệu nhiễu (noise):
    
    Trước hết, ta xem xét các thống kê mô tả của từng biến numeric trong bộ dữ liệu:
    
    ```
    ##      Ratings           Gross              Budget             Screens
    ##  Min.   :3.100   Min.   :     2470   Min.   :   70000   Min.   :   2
    ##  1st Qu.:5.800   1st Qu.: 10300000   1st Qu.: 9000000   1st Qu.: 449
    ##  Median :6.500   Median : 37400000   Median :28000000   Median :2777
    ##  Mean   :6.442   Mean   : 68066033   Mean   :47921730   Mean   :2209
    ##  3rd Qu.:7.100   3rd Qu.: 89350000   3rd Qu.:65000000   3rd Qu.:3372
    ##  Max.   :8.700   Max.   :643000000   Max.   :250000000   Max.   :4324
    ##                                      NA's   :1          NA's   :10
    ##    Sentiment           Views               Likes            Dislikes
    ##  Min.   :-38.00   Min.   :     698   Min.   :     1   Min.   :    0.0
    ##  1st Qu.:  0.00   1st Qu.:  623302   1st Qu.:  1776   1st Qu.:  105.5
    ##  Median :  0.00   Median : 2409338   Median :  6096   Median :  341.0
    ##  Mean   :  2.81   Mean   : 3712851   Mean   : 12732   Mean   :  679.1
    ##  3rd Qu.:  5.50   3rd Qu.: 5217380   3rd Qu.: 15248   3rd Qu.:  697.5
    ##  Max.   : 29.00   Max.   :32626778   Max.   :370552   Max.   :13960.0
    ##
    ##     Comments       AggregateFollowers
    ##  Min.   :    0.0   Min.   :    1066
    ##  1st Qu.:  248.5   1st Qu.:  183025
    ##  Median :  837.0   Median : 1052600
    ##  Mean   : 1825.7   Mean   : 3038193
    ##  3rd Qu.: 2137.0   3rd Qu.: 3694500
    ##  Max.   :38363.0   Max.   :31030000
    ##                    NA's   :35
    ```
    
    Ta nhận thấy:
    
    - **Biến “Budget”**: ngoài 1 giá trị bị khuyết ra thì có 3 giá trị dạng số thực (xem trong bộ dữ liệu), trong khi toàn bộ dữ liệu còn lại là số nguyên. Điều này cho thấy có sự không nhất quán về định dạng dữ liệu. Tuy nhiên, các giá trị này vẫn nằm trong khoảng hợp lý nên ta sẽ không xem là nhiễu.
    - **Biến “Sentiment”**: có giá trị trong khoảng $[-38, 29]$, cho thấy dữ liệu được đo trên một thang điểm có lẽ là $[-50, 50]$. Điều này cũng hợp lý.
    - **Biến “Dislikes”**: có 4 giá trị 0. Điều này hợp lý nhưng sẽ gây khó khăn nếu cần biến đổi biến.
    - **Biến “Comments”**: có 3 giá trị 0. Điều này hợp lý nhưng sẽ gây khó khăn nếu cần biến đổi biến.
    - Tất cả các biến đều có dữ liệu hợp lệ, ngoại trừ việc bị khuyết ở những biến đã nêu thì không phát hiện các giá trị bất thường vi phạm ràng buộc miền giá trị của từng biến.
    - Ta sẽ kiểm tra boxplot của từng biến để xem dữ liệu ngoại lai (outlier) thế nào:
    
    ![image.png](image.png)
    
    ![image.png](image%201.png)
    
    ![image.png](image%202.png)
    
    ![image.png](image%203.png)
    
    ![image.png](image%204.png)
    
    ![image.png](image%205.png)
    
    Từ hình ảnh boxplot, ta thấy:
    
    - Chỉ có biến **“Screens”** là không có outlier;
    - Biến **“Ratings”**: có 1 outlier;
    - Các biến còn lại: có nhiều outlier.
    
    ### Kiểm tra dữ liệu mâu thuẫn (conflict):
    
    Như đã nhận xét ở phần kiểm tra noise data, các biến đều có dữ liệu hợp lệ, không phát hiện các giá trị bất thường vi phạm ràng buộc miền giá trị của từng biến.
    Ta sẽ kiểm tra xem liệu có trường hợp mâu thuẫn về mặt logic thông thường hay không.
    
    - **Trường hợp không có ngân sách/ rạp chiếu mà có doanh thu:**
    
    ```
    ## [1] 0
    ```
    
    - **Trường hợp lượt Likes/ Dislikes > lượt Views:**
    
    ```
    ## [1] 0
    ```
    
    ```
    ## [1] 0
    ```
    
    - **Trường hợp phim ít ngân sách, ít rạp chiếu nhưng doanh thu lại cao bất thường:**
    
    ```
    ## [1] 0
    ```
    
    Ta kiểm tra suy rộng hơn một chút:
    
    - **Trường hợp phim được đánh giá tốt (Ratings cao và Sentiment tốt) nhưng doanh thu kém:**
    
    | # | Movie | Year | Ratings | Genre | Gross | Budget | Screens | Sequel | Sentiment | Views | Likes |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | Foxca... | 2014 | 7.1 | 9 | 1.21e6 | 2.40e7 | 66 | 1 | 6 | 6.69e6 | 8369 |
    | 2 | The G... | 2014 | 7.4 | 3 | 2.72e6 | 2e7 | 461 | 1 | 10 | 1.30e6 | 3306 |
    | 3 | The W... | 2014 | 7.2 | 3 | 4.19e6 | 2.25e7 | 320 | 1 | 7 | 3.28e6 | 4968 |
    | 4 | Wild | 2014 | 8.2 | 8 | 3.08e6 | 3.30e6 | 4 | 1 | 7 | 6.97e5 | 1023 |
    | 5 | Me an... | 2015 | 8.2 | 8 | 6.74e6 | 8e6 | 34 | 1 | 15 | 4.03e6 | 18398 |
    - **Trường hợp phim bị chê (Ratings thấp và Sentiment kém) nhưng doanh thu cao:**
    
    | # | Movie | Year | Ratings | Genre | Gross | Budget | Screens | Sequel | Sentiment | Views | Likes |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | Heave... | 2014 | 5.8 | 3 | 9.14e7 | 1.20e7 | 2417 | 1 | -9 | 175017 | 461 |
    | 2 | Trans... | 2014 | 5.8 | 1 | 2.45e8 | 2.10e8 | 4233 | 4 | 0 | 170909 | 791 |
    
    Ta thấy bộ dữ liệu không có dữ liệu mâu thuẫn. Có 7 quan trắc cho thấy mối quan hệ không phù hợp với logic thông thường, chẳng hạn như các bộ phim được đánh giá cao nhưng có doanh thu thấp, hoặc ngược lại.
    
- Kiểm tra phân phối
    
    Ta kiểm tra biểu đồ histogram của từng biến định lượng trong bộ dữ liệu:
    
    ![image.png](image%206.png)
    
    ![image.png](image%207.png)
    
    ![image.png](image%208.png)
    
    ![image.png](image%209.png)
    
    ![image.png](image%2010.png)
    
    ![image.png](image%2011.png)
    
    ![image.png](image%2012.png)
    
    ![image.png](image%2013.png)
    
    ![image.png](image%2014.png)
    
    ![image.png](image%2015.png)
    
    Từ các biểu đồ histogram, ta thấy:
    
    - Biến **“Ratings”** có vẻ như có phân phối chuẩn/ gần chuẩn.
    - Các biến **“Gross”**, **“Budget”**, **“Views”**, **“Likes”**, **“Dislikes”**, **“Comments”** và **“AggregateFollowers”** có phân phối lệch phải.
    - Biến **“Screens”** - số lượng rạp chiếu, có phân phối chia thành 2 cụm rõ rệt, điều này cho thấy có lẽ có một số phim chỉ phát hành hạn chế ở một vài nơi.
    - Biến **“Sentiment”** có vẻ như có phân phối gần chuẩn, nhưng dữ liệu tập trung về phần bên phải nhiều hơn.
    
    Từ phân phối của các biến, dự đoán rằng có lẽ ta cần phải thực hiện phép biến đổi biến trên một vài biến.
    
- Phân tích quan hệ giữa các biến
    - Ma trận tương quan giữa các biến
        
        ![image.png](image%2016.png)
        
    - Hoặc
    
    ![image.png](image%2017.png)
    
    - Từ ma trận tương quan giữa các biến, ta nhận thấy:
        - Biến **“Gross”** tương quan thuận và mạnh với biến **“Budget”** và **“Screens”** với hệ số tương quan lần lượt là $0.7$ và $0.57$. Điều này cho thấy ngân sách sản xuất và số lượng rạp chiếu ảnh hưởng nhiều nhất đến tổng doanh thu phim (xét về kiến thức chuyên môn thì điều này là hiển nhiên). Qua đây ta cũng dự đoán sự hiện diện không thể thiếu của 2 biến này trong mô hình dự báo doanh thu phim.
        - Biến **“Likes”** và biến **“Comments”** có hệ số tương quan $0.9$ cho thấy 2 biến này tương quan thuận rất chặt với nhau. Việc này sẽ dễ dẫn đến đa cộng tuyến.
        - Biến **“Views”** cũng có tương quan thuận và chặt với các biến **“Dislikes”**, **“Comments”** và **“Likes”** với hệ số tương quan $> 0.7$. Điều này cũng có thể dẫn đến đa cộng tuyến.
        - Các cặp biến **“Budget” - “Screens”**, **“Dislikes” - “Comments”**, **“Likes” - “Dislikes”** có hệ số tương quan trung bình $(\sim 0.5)$ cho thấy các cặp biến này có tương quan thuận với nhau với mức độ trung bình.
        - Các biến **“Sentiment”**, **“Likes”**, **“Dislikes”**, **“Views”**, **“Comments”** có tương quan rất yếu với biến **“Gross”**, nhất là biến **“Sentiment”** hầu như không có tương quan (hệ số tương quan $-0.04$), nên có thể dự đoán các biến này đóng góp rất ít vào mô hình dự báo doanh thu phim.
        
        *Sinh viên dùng thêm biểu đồ scatter plot để thấy xu hướng và mối quan hệ giữa các biến.*
        
        ### Kiểm tra đa cộng tuyến:
        
        ```
        ##           Ratings            Budget           Screens         Sentiment
        ##          1.342760          1.831254          1.689398          1.037939
        ##             Views             Likes          Dislikes          Comments
        ##          4.842820          6.750372          3.205007          7.208191
        ## AggregateFollowers
        ##          1.121843
        ```
        
        Từ hệ số phóng đại **VIF**, ta thấy có hiện tượng đa cộng tuyến, đặc biệt là ở biến **“Comments”** và **“Likes”**.
        

## 3. Chuẩn bị dữ liệu (Data Preparation)

Từ kết quả khám phá dữ liệu, ta lựa chọn phương pháp tiền xử lý dữ liệu phù hợp nhất với bộ dữ liệu nhằm mục đích xây dựng mô hình khớp dữ liệu nhất có thể

### 3.1 Lựa chọn biến (Select Data)

Ta nhận định biến **“Movie”** - tên phim, không có ý nghĩa khi phân tích theo tổng doanh thu phim. Do vậy, ta quyết định: loại bỏ nó ra khỏi bộ dữ liệu và cài đặt tên phim như là tên của từng dòng chứa dữ liệu quan trắc.

Do kết quả nhận được từ phân tích ma trận tương quan giữa các biến, ta cũng quyết định:

- **Loại bỏ biến “Comments”** khỏi bộ dữ liệu, giữ lại biến **“Likes”**, vì cả 2 biến này đều có hệ số tương quan với biến mục tiêu không chênh lệch là mấy nhưng biến **“Comments”** thì có 3 giá trị 0, trong khi biến **“Likes”** không có và số lượt thích thì có ý nghĩa hơn số bình luận.
- **Loại bỏ biến “Dislikes”** khỏi bộ dữ liệu vì giữa 3 biến **“Views”**, **“Likes”**, **“Dislikes”** thì biến **“Dislikes”** có 4 giá trị 0, trong khi 2 biến còn lại thì không có, và số lượt không thích thì không quan trọng bằng số lượt xem/ thích/ số người theo dõi.
- **Loại bỏ biến “Sentiment”** vì hầu như không có tương quan với biến mục tiêu **“Gross”** mà lại chứa nhiều giá trị ≤ 0.

Đây là bộ dữ liệu còn lại sau khi lựa chọn biến:

```
## tibble [231 × 10] (S3: tbl_df/tbl/data.frame)
##  $ Year              : Factor w/ 2 levels "2014","2015": 1 1 1 1 1 1 1 1 1 1 ...
##  $ Ratings           : num [1:231] 6.3 7.1 6.2 6.3 4.7 4.6 6.1 7.1 6.5 6.1 ...
##  $ Genre             : Factor w/ 11 levels "1","2","3","4",..: 7 1 1 1 7 3 7 1 9 7 ...
##  $ Gross             : num [1:231] 9.13e+03 1.92e+08 3.07e+07 1.06e+08 1.73e+07 2.90e+04 4.26e+07 ...
##  $ Budget            : num [1:231] 4.00e+06 5.00e+07 2.80e+07 1.10e+08 3.50e+06 5.00e+05 4.00e+07 ...
##  $ Screens           : num [1:231] 45 3306 2872 3470 2310 ...
##  $ Sequel            : Factor w/ 7 levels "1","2","3","4",..: 1 2 1 2 2 1 1 1 1 1 ...
##  $ Views             : num [1:231] 3280543 583289 304861 452917 3145573 ...
##  $ Likes             : num [1:231] 4632 3465 328 2429 12163 ...
##  $ AggregateFollowers: num [1:231] 1120000 12350000 483000 568000 1923800 ...
```

Ta còn lại 10 biến, bao gồm biến mục tiêu **“Gross”** và 9 biến giải thích.

### 3.2 Làm sạch dữ liệu (Clean Data)

- Xử lý dữ liệu trùng lặp (duplicated data)
    
    Bộ dữ liệu không có trùng lặp nên không xử lý
    
- Xử lý dữ liệu xung đột (conflict data)
    
    Từ các kết quả kiểm tra dữ liệu mâu thuẫn đã trình bày bên trên, ta thấy bộ dữ liệu không có dữ liệu mâu thuẫn.
    
    Có 7 quan trắc cho thấy mối quan hệ không phù hợp với logic thông thường, cụ thể là 5 bộ phim được đánh giá cao nhưng có doanh thu thấp, và 2 bộ phim ngược lại. Theo nhận định cá nhân tôi, đây không phải là conflict data, mà nó phản ánh có thể có các yếu tố khác như hoạt động marketing, sức hút của dàn diễn viên, thương hiệu của nhà sản xuất,... tác động tới doanh thu phim.
    
- Xử lý dữ liệu nhiễu (noise data)
    
    Vì biến **“Ratings”** chỉ có 1 outlier nên ta quyết định loại bỏ quan trắc có ngoại lai của biến này khỏi bộ dữ liệu để hầu nhận được mô hình có thể dự báo tốt hơn.
    
    ```
    ## [1] 230 10
    ```
    
    Sau khi loại bỏ outlier của biến **“Ratings”** thì bộ dữ liệu còn lại 230 quan trắc. Ta kiểm tra lại bằng biểu đồ boxplot:
    
    ![image.png](image%2018.png)
    
    Ta thấy biến **“Ratings”** đã hết ngoại lai.
    
    Còn các biến còn lại có quá nhiều ngoại lai mà lại có phân phối lệch nên ta tạm thời chưa xử lý chúng vì có thể chúng không thực sự là ngoại lai. Ta sẽ quay lại kiểm tra ngoại lai sau khi biến đổi biến (nếu cần thiết phải biến đổi).
    
    Qua đây cũng phần nào dự đoán rằng có lẽ ta cần phải thực hiện phép biến đổi biến để đưa các biến này về dạng chuẩn/gần chuẩn.
    
- Xử lý dữ liệu khuyết (missing data)
    
    Trước hết, ta kiểm tra xem có xảy ra missing đồng thời nhiều biến trong cùng 1 quan trắc hay không:
    
    ```
    ## Số quan trắc missing đồng thời cả 3 biến Budget, Screens và AggregateFollowers: 0
    ```
    
    ```
    ## Số quan trắc missing đồng thời cả biến Budget và biến Screens: 0
    ```
    
    ```
    ## Số quan trắc missing đồng thời cả biến Budget và biến AggregateFollowers: 0
    ```
    
    ```
    ## Số quan trắc missing đồng thời cả biến Screens và biến AggregateFollowers: 2
    ```
    
    Ta thấy có 2 quan trắc missing đồng thời cả biến **Screens** và biến **AggregateFollowers**, ta quyết định loại bỏ 2 quan trắc này để tránh sai lệch dữ liệu:
    
    ```
    ## [1] 228 10
    ```
    
    Khi đó, bộ dữ liệu còn lại 228 quan trắc.
    
    ### Xử lý missing value cho biến “Budget”: (số lượng: 1)
    
    Ta ước lượng giá trị điền khuyết cho biến **“Budget”** bằng mô hình hồi quy tuyến tính sử dụng các biến **“Gross”**, **“Screens”**, và **“Ratings”** dựa trên các quan tắc có cùng thể loại phim:
    
    ```
    ##          1
    ## 7088618
    ```
    
    **Kết quả điền khuyết:**
    
    | Year | Ratings | Genre | Gross | Budget | Screens | Sequel | Views | Likes | AggregateFollowers |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 2014 | 7 | 8 | 2820000 | 7088618. | 432 | 1 | 10747 | 4 | *Dữ liệu khuyết* |
    
    ```
    ## # A tibble: 1 × 10
    ##   Year  Ratings Genre     Gross   Budget Screens Sequel Views Likes
    ##   <fct>   <dbl> <fct>     <dbl>    <dbl>   <dbl> <fct>  <dbl> <dbl>
    ## 1 2014        7 8       2820000 7088618.     432 1      10747     4
    ## # i 1 more variable: AggregateFollowers <dbl>
    ```
    
    ### Xử lý missing value cho biến “Screens”: (số lượng còn lại: 8)
    
    Ta ước lượng 8 giá trị điền khuyết cho biến **“Screens”** bằng mô hình hồi quy tuyến tính sử dụng các biến **“Budget”**, **“Gross”**, **“Views”**, và **“AggregateFollowers”** dựa trên các quan trắc có cùng thể loại phim và cùng năm phát hành:
    
    ```
    ##    Row Predicted_Screens
    ## 1    6               192
    ## 11  25               229
    ## 12  33              2320
    ## 13  39              1929
    ## 14  67               283
    ## 15  84               928
    ## 16  95               259
    ## 17 127               584
    ```
    
    **Kết quả điền khuyết:**
    
    ```
    ## # A tibble: 8 × 10
    ##    Year  Ratings Genre     Gross   Budget Screens Sequel   Views  Likes
    ##    <fct>   <dbl> <fct>     <dbl>    <dbl>   <dbl> <fct>    <dbl>  <dbl>
    ## 1 2014      4.6 3         29000   500000     192 1        91137    112
    ## 2 2014      7   3          9840  1000000     229 1         7128      1
    ## 3 2014      6.3 1      55900000 70000000    2320 1      9149892  26427
    ## 4 2014      5.7 1          8690  4500000    1929 1       735551    636
    ## 5 2014      5.7 8          8300  2400000     283 1      1222921   5553
    ## 6 2014      5.2 8         35700   600000     928 1      5403836 187162
    ## 7 2014      7.5 8         37400  5000000     259 1       827239   3221
    ## 8 2014      5   3       2820000 12000000     584 1       309610    729
    ## # i 1 more variable: AggregateFollowers <dbl>
    ```
    
    ### Xử lý missing value cho biến “AggregateFollowers”: (số lượng còn lại: 33)
    
    ```
    ##    Row Predicted_AggregateFollowers
    ## 1    26                   168700
    ## 2    44                  2043250
    ## 3    49                   184100
    ## 4    55                  1658900
    ## 5    59                   130000
    ## 6    62                  1480000
    ## 7    65                  1480000
    ## 8    69                  1480000
    ## 9    76                  1658900
    ## 10   80                  2043250
    ## 11   94                   168700
    ## 12   98                   184100
    ## 13  109                  2613000
    ## 14  111                   184100
    ## 15  115                  2043250
    ## 16  128                   168700
    ## 17  137                  1280000
    ## 18  143                   184100
    ## 19  155                   184100
    ## 20  157                  1658900
    ## 21  216                  1630000
    ## 22  217                  1630000
    ## 23  218                   946000
    ## 24  219                  4599000
    ## 25  220                  2145500
    ## 26  221                  1630000
    ## 27  222                  2145500
    ## 28  223                  2145500
    ## 29  224                  1630000
    ## 30  225                       NA
    ## 31  226                     2454
    ## 32  227                  2145500
    ## 33  228                     2454 
    ```
    
    Kết quả điền khuyết:
    
    ```
    ## # A tibble: 33 × 10
    ##     Year  Ratings Genre      Gross   Budget Screens Sequel   Views  Likes
    ##     <fct>   <dbl> <fct>      <dbl>    <dbl>   <dbl> <fct>    <dbl>  <dbl>
    ##  1 2014      6.7 9        6370000  3000000     382 1      2902492   9522
    ##  2 2014      6.1 10        104000  1000000       3 1        99427     47
    ##  3 2014      5.8 3       91400000 12000000    2417 1       175017    461
    ##  4 2014      6.7 8        8090000 20000000     645 1      1167941   2651
    ##  5 2014      6.9 15      14700000  2000000       4 1         4877      6
    ##  6 2014      7.2 1       43000000 20000000    2589 1      4846645  14722
    ##  7 2014      6.5 1         129000 25000000      28 1       289922    143
    ##  8 2014      6.4 1      127000000 40000000    3173 1      1142964   2346
    ##  9 2014      6.3 8        4010000  5000000     255 1       446576    659
    ## 10 2014      7.9 10      32300000  8500000    2766 1      6082510  12522
    ## # i 23 more rows
    ## # i 1 more variable: AggregateFollowers <dbl>
    ```
    

### 3.3 Xây dựng dữ liệu (Construct Data)

- Biến đổi biến (transformation)
    
    ### Dự đoán transformation bằng thống kê mô tả
    
    Từ biểu đồ tương quan và từ hình ảnh boxplot, histogram của các biến trong bộ dữ liệu, ta thấy có 4 nhóm biến sau đây:
    
    - **Biến “Year”, “Genre” và “Sequel”**: là factor nên ta sẽ không biến đổi gì.
    - **Biến “Ratings”**: có vẻ như có phân phối chuẩn/ gần chuẩn nên có lẽ không cần biến đổi, ta sẽ kiểm tra lại bằng thống kê suy diễn.
    - **Các biến “Gross”, “Budget”, “Views”, “Likes” và “AggregateFollowers”**: có phân phối lệch phải nên có lẽ sẽ cần phép biến đổi log, hoặc lấy căn ($1/\lambda$), hoặc lấy mũ nhỏ hơn 1. Đặc biệt, hai biến **“Gross”** và **“Budget”** thể hiện số tiền có bậc độ lớn nên có lẽ sẽ cần phép biến đổi log.
    - **Biến “Screens”**: có vẻ như có phân phối lệch phải và chia thành 2 cụm rõ rệt, có lẽ ta cần xem xét một vài phép biến đổi.
    
    ### Xác định transformation bằng thống kê suy diễn
    
    - **Kiểm định Likelihood ratio để xác định liệu có cần biến đổi biến hay không.**
        - **Test 2 - No transformation is needed** (ứng với 1)
            - Giả thuyết H0: không cần biến đổi bất cứ biến nào cả.
            - Đối thuyết HA: cần phải biến đổi ít nhất một biến.
        - **Test 1 - Log transformation parameter is equal 0** (ứng với $\lambda = 0$)
            - Giả thuyết H0: cần biến đổi log-transformation cho tất cả các biến.
            - Đối thuyết HA: tồn tại ít nhất một biến không cần biến đổi log-transformation.
    
    **Cách khác:** cần phải biến đổi ít nhất một biến trong mô hình, với khả năng sai lầm $5\%, p\text{-value} = 2.22e\text{-}16 < \alpha = 0.05$.
    
    ```
    ## p-value = 2.22e-16 < alpha = 0.05 nên ta bác bỏ H0 với khả năng sai lầm 5 %.
    ## Hay nói cách khác: cần phải biến đổi ít nhất một biến trong mô hình không cần biến đổi log transformation, với khả năng sai lầm 5%
    ```
    
    - **Kiểm định Likelihood ratio để kiểm tra xem có cần sử dụng giá trị $\lambda$  cho phép biến đổi biến hay không.**
        - Giả thuyết H0: KHÔNG sử dụng power $\lambda$ cho phép transform.
        - Đối thuyết HA: sử dụng power $\lambda$ cho phép transform.
    
    ```
    ## p-value = 2.22e-16 < alpha = 0.05 nên ta bác bỏ H0 với khả năng sai lầm 5 %.
    ## Hay nói cách khác: sử dụng power lambda cho phép transform.
    ```
    
    **Tổng hợp các kết quả:** cần phải biến đổi ít nhất 1 biến và không biến đổi log cho tất cả các biến mà sử dụng power $\lambda$
    
    ### Xác định biến đổi như thế nào
    
    Trước hết, ta kiểm tra xem biến **“Ratings”** có phân phối chuẩn/ gần chuẩn gì không:
    
    ```
    ## [1] 0.08523092
    ```
    
    Ta thấy $p\text{-value} = 0.08523092 > \alpha = 0.05$ nên có thể cho rằng biến **“Ratings”** có phân phối chuẩn với khả năng sai lầm 5%.
    
    **Vậy ta quyết định:**
    
    - **Biến “Ratings”**: giữ nguyên.
    - **Các biến factor “Year”, “Genre” và “Sequel”**: giữ nguyên.
    - **Các biến “Gross”, “Budget”, “Screens”, “Likes”, “Dislikes” và “AggregateFollowers”**: ta sẽ tiến hành biến đổi.
    
    ### Biến đổi “Gross”:
    
    Ta dùng phương pháp vét cạn lần lượt kiểm tra p-value của Shapiro-Wilk test tương ứng để tìm ra số mũ $\lambda$ dùng để biến đổi biến **“Gross”**:
    
    ```
    ## [1] "0.00582293754621291" "3"                  "cube root"
    ```
    
    →  Ta chọn biến đổi **Gross** thành $\sqrt[3]{\text{Gross}}$.
    
    - **Biến đổi “Budget”**:
    
    ```
    ## [1] "0.0739008510061309" "5"                  "square root"
    ```
    
    → Ta chọn biến đổi **Budget** thành $\sqrt[5]{Budget}$.
    
    - **Biến đổi “Screens”**:
    
    ```
    ## [1] "6.62323469385674e-12" "1.5"                "exponent"
    ```
    
    → Ta chọn biến đổi **Screens** thành $Screens^{3/2}$.
    
    - **Biến đổi “Views”**:
    
    ```
    ## [1] "0.113934834463708" "4"                  "square root"
    ```
    
    → Ta chọn biến đổi **Views** thành $\sqrt[4]{Views}$.
    
    - **Biến đổi “Likes”**:
    
    ```
    ## [1] "0.00495730572343909" "5"                  "square root"
    ```
    
    → Ta chọn biến đổi **Likes** thành $\sqrt[5]{Likes}$.
    
    - **Biến đổi “AggregateFollowers”**:
    
    ```
    ## [1] "0.0235987182977128" "5"                  "square root"
    ```
    
    → Ta chọn biến đổi **AggregateFollowers** thành  $\sqrt[5]{AggregateFollowers}$
    
    ### Kiểm tra lại bằng phương pháp Box-Cox:
    
    ```
    ## Lambda cho các biến Gross, Budget, Screens, Views, Likes, AggregateFollowers lần lượt là:
    ## 0.3 0.2 0.5 0.3 0.2 0.2
    ```
    
    Ta thấy kết quả của phương pháp Box-Cox gần khớp với các giá trị $\lambda$ mà ta đã chọn cho các biến **“Gross”**, **“Budget”**, **“Views”**, **“Likes”**, **“AggregateFollowers”** (chênh lệch nhỏ, có lẽ do Box-Cox làm tròn số), và không khớp với giá trị $\lambda$ mà ta đã chọn cho biến **“Screens”**.
    
    Điều này dễ hiểu vì 5 biến đầu có phân phối lệch phải, còn biến **“Screens”** thì có phân phối phức tạp hơn, gồm nhiều đỉnh, chia thành 2 cụm rõ rệt. Nên có lẽ ta nên thử cả 2 cách biến đổi này và cùng với cách giữ nguyên biến **“Screens”** ban đầu.
    
    **Ta kiểm tra lại histogram của các biến sau khi biến đổi**
    
    - Theo phương pháp đề xuất
        
        ![image.png](image%2019.png)
        
    - Theo Box-Cox
        
        ![image.png](image%2020.png)
        
        Qua các biểu đồ histogram, ta thấy phân phối của các biến đã về gần phân phối Gauss hơn, dự báo một kết quả tốt đẹp hơn.
        
        **Nhắc lại:** ở phần *Xử lý dữ liệu ngoại lai*, ta vẫn giữ lại nhiều giá trị ngoại lai mà chưa xử lý. Bây giờ, sau khi thực hiện biến đổi biến, ta kiểm tra lại xem các biến có còn ngoại lai hay không.
        
        Hình ảnh boxplot của các biến có ngoại lai trước và sau khi biến đổi biến:
        
        ![image.png](image%2021.png)
        
        (Mất nội dung từ khúc sau vì video bị đứng)
        
        Bỏ đi 3 outline từ các biến ít
        
- Tạo biến mới
    
    (Phần này video bị mất - Chép lại theo lồng tiếng)
    
    Trong quá trình bộ dữ liệu, các biến tương tác với nhau và gây ảnh hưởng tới mô hình, có thể thực hiện các kiểm định như kiểm định ANOVA,… để xem giữa các biến có tương tác với nhau và tạo ra biến mới, và gây ảnh hưởng tới tổng danh thu, dẫn đến việc thêm biến đó vô, vậy ta phải xem xét sự tương tác của các biến
    

## 4. Xây dựng mô hình (Modeling)

- **Tập Train:** 80% dữ liệu;
- **Tập Validation:** 20% dữ liệu.

```
## Kích thước tập Train: 182 dòng, 10 cột.
```

```
## Kích thước tập validation: 42 dòng, 10 cột.
```

Như vậy, bộ Train có 182 quan trắc và bộ Validation có 42 quan trắc.

- 4.1 Xây dựng mô hình với đầy đủ tất cả các biến
    
    Khai báo mô hình `full_model`:
    
    $full\_model = lm(Gross \sim Year + Ratings + Genre + Budget + Screens + Sequel + Views + Likes + AggregateFollowers)$
    
    ### Kiểm tra đa cộng tuyến:
    
    | Biến | GVIF | Df | $GVIF^{1/(2 \cdot Df)}$ |
    | --- | --- | --- | --- |
    | Year | 1.317910 | 1 | 1.148003 |
    | Ratings | 1.359499 | 1 | 1.165975 |
    | Genre | 3.226044 | 8 | 1.075950 |
    | Budget | 2.672270 | 1 | 1.634708 |
    | Screens | 2.272584 | 1 | 1.507509 |
    | Sequel | 2.646757 | 6 | 1.084492 |
    | Views | 4.481501 | 1 | 2.116956 |
    | Likes | 4.786241 | 1 | 2.187748 |
    | AggregateFollowers | 1.401471 | 1 | 1.183838 |
    
    Ta thấy hệ số phóng đại **VIF** của tất cả các biến trong mô hình đều < 2.2 nên không có hiện tượng đa cộng tuyến.
    
    **Kiểm định ý nghĩa của mô hình (Kiểm định Fisher toàn phần)**
    
    - Giả thuyết $H_0 : \beta_0 = \beta_1 = \beta_2 = \dots = \beta_9 = 0$
    - Đối thuyết $H_A : \exists \beta_i \neq 0$
    - Mức ý nghĩa: $\alpha = 0.05$
    
    ```
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)        -1.334e+08  3.063e+07  -4.356 2.36e-05 ***
    ## Year2015            1.563e+06  8.837e+06   0.177 0.859802
    ## Ratings             1.903e+07  4.446e+06   4.279 3.22e-05 ***
    ## Genre2              1.701e+07  1.764e+07   0.964 0.336382
    ## Genre3              1.012e+05  1.212e+07   0.008 0.993345
    ## Genre6             -2.510e+07  2.900e+07  -0.866 0.388010
    ## Genre8             -3.229e+06  1.160e+07  -0.278 0.781056
    ## Genre9              1.241e+07  1.758e+07   0.706 0.481164
    ## Genre10            -6.746e+06  1.787e+07  -0.378 0.706240
    ## Genre12             4.732e+06  1.546e+07   0.306 0.759908
    ## Genre15             2.649e+07  2.239e+07   1.183 0.238604
    ## Budget              7.154e-01  9.972e-02   7.174 2.56e-11 ***
    ## Screens             1.279e+04  3.621e+03   3.532 0.000539 ***
    ## Sequel2             3.300e+07  1.229e+07   2.684 0.008039 ** ## Sequel3             1.130e+07  2.114e+07   0.534 0.593747
    ## Sequel4             1.331e+07  3.600e+07   0.370 0.712150
    ## Sequel5            -3.638e+07  2.543e+07  -1.431 0.154434
    ## Sequel6            -6.935e+06  5.321e+07  -0.130 0.896468
    ## Sequel7             5.789e+07  3.776e+07   1.533 0.127197
    ## Views               6.712e-01  1.768e+00   0.380 0.704786
    ## Likes              -1.343e+02  6.679e+02  -0.201 0.840847
    ## AggregateFollowers  2.582e+00  9.076e-01   2.845 0.005028 ** ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ##
    ## Residual standard error: 47690000 on 160 degrees of freedom
    ## Multiple R-squared:  0.7148, Adjusted R-squared:  0.6773
    ## F-statistic: 19.09 on 21 and 160 DF,  p-value: < 2.2e-16
    ```
    
    **Nhận xét:** $p - value = 2.2 \times 10^{-16} < \alpha = 0.05$ nên ta có đủ bằng chứng chống lại giả thuyết $H_0$ với khả năng sai lầm 5%. Hay nói cách khác: mô hình `full_model` chấp nhận được với khả năng sai lầm 5%.
    
    ### Kiểm định ý nghĩa của từng biến đối với mô hình (Kiểm định student)
    
    - Giả thuyết ứng với mỗi biến i: $H_0 : \beta_i = 0$
    - Đối thuyết ứng với mỗi biến i: $H_A : \beta_i \neq 0$
    
    **Nhận xét:** dựa vào giá trị của các $p\text{-value}$ trong các kiểm định student bên trên, ta thấy chỉ có 6 biến **“Ratings”**, **“Budget”**, **“Screens”**, **“AggregateFollowers”** và **“Sequel”** loại 2 (do biến **“Sequel”** là định tính) là có ý nghĩa (do $p\text{-value} < \alpha = 0.05$), trong đó biến **“Sequel”** là có ý nghĩa ít nhất, còn các biến còn lại không có ý nghĩa với mô hình.
    
- 4.2 Lựa chọn mô hình khớp dữ liệu
    
    Từ các kết quả kiểm định, ta thấy mô hình `full_model` có ý nghĩa, và các biến **“Ratings”**, **“Budget”**, **“Screens”**, **“AggregateFollowers”**, **“Sequel”** có ý nghĩa với mô hình nên ta thử dùng phương pháp stepwise backward với tiêu chuẩn AIC để chọn ra mô hình tốt hơn/ ít biến giải thích hơn:
    
    ```
    ## Start:  AIC=6456.12
    ## Gross ~ Year + Ratings + Genre + Budget + Screens + Sequel +
    ##     Views + Likes + AggregateFollowers
    ##
    ##                      Df  Sum of Sq       RSS    AIC
    ## - Genre               8 1.0642e+16 3.7447e+17 6445.4
    ## - Year                1 7.1168e+13 3.6390e+17 6454.2
    ## - Likes               1 9.1994e+13 3.6392e+17 6454.2
    ## - Views               1 3.2757e+14 3.6416e+17 6454.3
    ## <none>                             3.6383e+17 6456.1
    ## - Sequel              6 2.7777e+16 3.9161e+17 6457.5
    ## - AggregateFollowers  1 1.8401e+16 3.8223e+17 6463.1
    ## - Screens             1 2.8364e+16 3.9219e+17 6467.8
    ## - Ratings             1 4.1644e+16 4.0547e+17 6473.8
    ## - Budget              1 1.1704e+17 4.8087e+17 6504.9
    ```
    
    ```
    ## Step:  AIC=6445.37
    ## Gross ~ Year + Ratings + Budget + Screens + Sequel + Views + 
    ##     Likes + AggregateFollowers
    ## 
    ##                      Df  Sum of Sq       RSS    AIC
    ## - Year                1 4.3117e+13 3.7451e+17 6443.4
    ## - Likes               1 1.0839e+14 3.7458e+17 6443.4
    ## - Views               1 3.2696e+14 3.7480e+17 6443.5
    ## <none>                             3.7447e+17 6445.4
    ## - Sequel              6 2.5722e+16 4.0019e+17 6445.5
    ## - AggregateFollowers  1 1.8787e+16 3.9326e+17 6452.3
    ## - Screens             1 3.5389e+16 4.0986e+17 6459.8
    ## - Ratings             1 4.5171e+16 4.1964e+17 6464.1
    ## - Budget              1 1.4093e+17 5.1540e+17 6501.5
    ```
    
    ```
    ## Step:  AIC=6443.39
    ## Gross ~ Ratings + Budget + Screens + Sequel + Views + Likes +
    ##     AggregateFollowers
    ##
    ##                      Df  Sum of Sq       RSS    AIC
    ## - Likes               1 1.4117e+14 3.7465e+17 6441.5
    ## - Views               1 3.3675e+14 3.7485e+17 6441.6
    ## <none>                             3.7451e+17 6443.4
    ## - Sequel              6 2.5742e+16 4.0026e+17 6443.5
    ## - AggregateFollowers  1 1.9410e+16 3.9392e+17 6450.6
    ## - Screens             1 3.6082e+16 4.1060e+17 6458.1
    ## - Ratings             1 4.5193e+16 4.1971e+17 6462.1
    ## - Budget              1 1.4257e+17 5.1708e+17 6500.1
    ```
    
    ```
    ## Step:  AIC=6441.46
    ## Gross ~ Ratings + Budget + Screens + Sequel + Views +
    ##     AggregateFollowers
    ##
    ##                      Df  Sum of Sq       RSS    AIC
    ## - Views               1 2.5020e+14 3.7490e+17 6439.6
    ## <none>                             3.7465e+17 6441.5
    ## - Sequel              6 2.5653e+16 4.0031e+17 6441.5
    ## - AggregateFollowers  1 1.9285e+16 3.9394e+17 6448.6
    ## - Screens             1 3.6106e+16 4.1076e+17 6456.2
    ## - Ratings             1 4.6304e+16 4.2096e+17 6460.7
    ## - Budget              1 1.4361e+17 5.1826e+17 6498.5
    ```
    
    ```
    ## Step:  AIC=6439.58
    ## Gross ~ Ratings + Budget + Screens + Sequel + AggregateFollowers
    ##
    ##                      Df  Sum of Sq       RSS    AIC
    ## - Sequel              6 2.5430e+16 4.0033e+17 6439.5
    ## <none>                             3.7490e+17 6439.6
    ## - AggregateFollowers  1 2.0491e+16 3.9540e+17 6447.3
    ## - Screens             1 3.9511e+16 4.1442e+17 6455.8
    ## - Ratings             1 4.6129e+16 4.2103e+17 6458.7
    ## - Budget              1 1.4395e+17 5.1886e+17 6496.7
    ```
    
    ```
    ## Step:  AIC=6439.53
    ## Gross ~ Ratings + Budget + Screens + AggregateFollowers
    ##
    ##                      Df  Sum of Sq       RSS    AIC
    ## <none>                             4.0033e+17 6439.5
    ## - AggregateFollowers  1 3.6735e+16 4.3707e+17 6453.5
    ## - Ratings             1 4.4918e+16 4.4525e+17 6456.9
    ## - Screens             1 4.5863e+16 4.4620e+17 6457.3
    ## - Budget              1 1.9229e+17 5.9263e+17 6508.9
    ```
    
    Vậy mô hình `short_model` ta thu được là:
    
    $short\_model = lm(Gross \sim Ratings + Budget + Screens + AggregateFollowers)$
    
    Ta thấy `short_model` đúng với những gì ta vừa nhận xét ở mục 4.1 bên trên.
    
    Ta tiến hành kiểm định lại ý nghĩa của mô hình `short_model` này và ý nghĩa của các biến trong mô hình (như đã thực hiện với `full_model` bên trên):
    
    Kết quả chi tiết của mô hình `short_model`:
    
    ```
    ##
    ## Call:
    ## lm(formula = Gross ~ Ratings + Budget + Screens + AggregateFollowers,
    ##     data = train)
    ##
    ## Residuals:
    ##        Min         1Q     Median         3Q        Max
    ## -126928343  -24311251   -2231699   18625532  215329895
    ##
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)        -1.263e+08  2.695e+07  -4.688 5.49e-06 ***
    ## Ratings             1.799e+07  4.036e+06   4.456 1.47e-05 ***
    ## Budget              7.572e-01  8.212e-02   9.221  < 2e-16 ***
    ## Screens             1.406e+04  3.122e+03   4.503 1.21e-05 ***
    ## AggregateFollowers  3.183e+00  7.899e-01   4.030 8.27e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ##
    ## Residual standard error: 47560000 on 177 degrees of freedom
    ## Multiple R-squared:  0.6861,	Adjusted R-squared:  0.679
    ## F-statistic: 96.74 on 4 and 177 DF,  p-value: < 2.2e-16
    ```
    
    Dựa trên kết quả này, ta thấy tất cả các biến còn lại trong mô hình (`Ratings`, `Budget`, `Screens`, `AggregateFollowers`) đều có ý nghĩa thống kê rất cao $p\text{-value} < 0.001$. Chỉ số **Adjusted R-squared** đạt 0.679, cho thấy mô hình giải thích được khoảng 67.9% sự biến thiên của doanh thu.
    
    ### Nhận xét:
    
    - $p - value = 2.2 \times 10^{-16} < \alpha = 0.05$ nên ta bác bỏ giả thuyết $H_0$ với khả năng sai lầm 5%. Hay nói cách khác: mô hình `short_model` chấp nhận được với khả năng sai lầm 5%.
    - Trong đó, tất cả các biến **“Ratings”**, **“Budget”**, **“Screens”**, **“AggregateFollowers”** đều có ý nghĩa, biến **Sequel** đã bị loại khỏi mô hình. Điều này hoàn toàn khớp với kết quả của phương pháp stepwise lẫn kết quả của `full_model` bên trên.
    - So sánh $R^2_{adj}$ của `short_model` và `full_model`:
    
    ```
    ## R^2 hiệu chỉnh của full_model = 0.6773228
    ```
    
    ```
    ## R^2 hiệu chỉnh của short_model = 0.6790467
    ```
    
    Ta thấy `short_model` có hệ số $R^2_{adj}$ tốt hơn `full_model` 0.2%.
    
    ### Kiểm tra sai số của mô hình:
    
    - **Kiểm tra bằng thống kê mô tả:**
        
        Đồ thị của sai số $\epsilon$:
        
        ![image.png](image%2022.png)
        
        Nhận xét từ đồ thị:
        
        - **Đồ thị Residuals vs Fitted** (Kiểm tra tính tuyến tính và đồng phương sai): cho thấy các điểm không phân bổ ngẫu nhiên quanh 0 và đường hồi quy có dạng cong nên khó mà xác định được tính đồng nhất của phương sai và có vẻ như giả định tuyến tính cũng không thỏa.
        - **Đồ thị Q-Q Residuals** (kiểm tra tính chuẩn của sai số): cho thấy hầu hết giá trị thặng dư đều gần với đường phân phối chuẩn, tuy nhiên ở 2 đầu, đặc biệt là đầu phải, có rất nhiều ngoại lai và lệch xa đường phân phối chuẩn nên có vẻ như sai số không có phân phối chuẩn.
        - **Đồ thị Scale-Location** (kiểm tra tính ổn định của phương sai): cho thấy đường màu đỏ tăng rõ rệt, phần lớn các giá trị thặng dư tập trung ở bên cạnh trái và còn có rất nhiều ngoại lai ở giữa và bên cạnh phải nên có vẻ như phương sai sẽ không ổn định.
        - **Đồ thị Residuals vs Leverage** (xác định các quan trắc có ảnh hưởng lớn): cho thấy có nhiều outlier trong mô hình.
    
    Tuy nhiên, những nhận định này chỉ là phỏng đoán dựa trên hình ảnh thống kê mô tả của `short_model`, ta cần kiểm tra lại các nhận định này bằng các kiểm định trong thống kê suy diễn.
    
    - **Kiểm tra bằng thống kê suy diễn:**
        - **Để kiểm tra tính chuẩn của sai số mô hình, ta thực hiện Kiểm định Shapiro-Wilk:**
            - Giả thuyết $H_0 : \epsilon \sim N$
            - Đối thuyết $H_A : \epsilon \not\sim N$
        
        ```
        ## p-value = 1.864809e-07 < alpha = 0.05 nên ta bác bỏ H0 với khả năng sai lầm 5 %.
        ## Hay nói cách khác: sai số của mô hình không có phân phối chuẩn.
        ```
        
        - **Để kiểm tra trung bình của sai số có bằng 0 hay không, ta thực hiện Kiểm định student:**
            - Giả thuyết $H_0 : E(\epsilon) = 0$
            - Đối thuyết $H_A : E(\epsilon) \neq 0$
        
        ```
        ## p-value = 1 nên ta chấp nhận H0 với khả năng sai lầm 5 %.
        ## Hay nói cách khác: E(sai số) = 0 với khả năng sai lầm 5%.
        ```
        
        - **Để kiểm tra tính ổn định của phương sai, ta thực hiện Kiểm định Non-constant Variance Score:**
            - Giả thuyết $H_0$: phương sai ổn định
            - Đối thuyết $H_A$: phương sai không ổn định
        
        ```
        ## p-value = 2.543528e-17 < alpha = 0.05 nên ta bác bỏ H0 với khả năng sai lầm 5 %.
        ## Hay nói cách khác: mô hình có phương sai không ổn định.
        ```
        
    
    **Kết luận:** sai số mô hình `short_model` không thỏa tính chuẩn và phương sai không ổn định.
    
- 4.3 Mô hình biến đổi biến
    
    Cả 2 mô hình `short_model` và `full_model` có $R^2_{adj}$ xấp xỉ nhau (khoảng 67%) và quan trọng hơn hết là nó không thỏa giá định hồi quy tuyến tính.
    
    → ta cần thực hiện các phép biến đổi biến ở mục 3.3 để đưa mô hình về dạng thỏa các giả định của hồi quy tuyến tính.
    
    Do ta có 2 phương pháp biến đổi biến: phương pháp đề xuất và phương pháp theo Box-Cox, kết hợp với việc nhận định về phân phối phức tạp của biến **“Screens”**, ta thành lập 4 mô hình như sau:
    
    - **full_model_mytransform** : là mô hình sử dụng các biến đổi biến đề xuất:
        
        $lm(I(Gross^{1/3}) \sim Year + Ratings + Genre + I(Budget^{1/5}) + I(Screens^{3/2}) + Sequel + I(Views^{1/4}) + I(Likes^{1/5}) + I(AggregateFollowers^{1/5}))$
        
    - **full_model_BCtransform** : là mô hình sử dụng các biến đổi biến theo Box-Cox:
        
        $lm(I(Gross^{0.3}) \sim Year + Ratings + Genre + I(Budget^{0.2}) + I(Screens^{0.5}) + Sequel + I(Views^{0.3}) + I(Likes^{0.2}) + I(AggregateFollowers^{0.2}))$
        
    - **full_model_mytransform2** : là mô hình sử dụng các biến đổi biến đề xuất, nhưng giữ nguyên “Screens”:
        
        $lm(I(Gross^{1/3}) \sim Year + Ratings + Genre + I(Budget^{1/5}) + Screens + Sequel + I(Views^{1/4}) + I(Likes^{1/5}) + I(AggregateFollowers^{1/5}))$
        
    - **full_model_BCtransform2** : là mô hình sử dụng các biến đổi biến theo Box-Cox, nhưng giữ nguyên “Screens”:
        
        $lm(I(Gross^{0.3}) \sim Year + Ratings + Genre + I(Budget^{0.2}) + Screens + Sequel + I(Views^{0.3}) + I(Likes^{0.2}) + I(AggregateFollowers^{0.2}))$
        
    
    ### Kiểm tra đa cộng tuyến cho các mô hình:
    
    **1. Đối với mô hình full_model_mytransform:**
    
    | Biến | GVIF | Df | GVIF^{1/(2 \cdot Df)} |
    | --- | --- | --- | --- |
    | Year | 1.297156 | 1 | 1.138927 |
    | Ratings | 1.299187 | 1 | 1.139819 |
    | Genre | 3.792399 | 8 | 1.086881 |
    | I(Budget^(1/5)) | 3.260040 | 1 | 1.805558 |
    | I(Screens^(3/2)) | 2.901954 | 1 | 1.703512 |
    | Sequel | 2.444200 | 6 | 1.077320 |
    | I(Views^(1/4)) | 10.082830 | 1 | 3.175347 |
    | I(Likes^(1/5)) | 10.885884 | 1 | 3.299376 |
    | I(AggregateFollowers^(1/5)) | 1.498636 | 1 | 1.224188 |
    
    **2. Đối với mô hình full_model_BCtransform:**
    
    | Biến | GVIF | Df | $GVIF^{1/(2 \cdot Df)}$ |
    | --- | --- | --- | --- |
    | Year | 1.274660 | 1 | 1.129009 |
    | Ratings | 1.335856 | 1 | 1.155792 |
    | Genre | 3.747413 | 8 | 1.086071 |
    | I(Budget^(0.2)) | 2.760688 | 1 | 1.661532 |
    | I(Screens^(0.5)) | 2.183427 | 1 | 1.477642 |
    | Sequel | 2.373678 | 6 | 1.074695 |
    | I(Views^(0.3)) | 9.208644 | 1 | 3.034575 |
    | I(Likes^(0.2)) | 10.096011 | 1 | 3.177422 |
    | I(AggregateFollowers^(0.2)) | 1.495638 | 1 | 1.222963 |
    
    **3. Đối với mô hình full_model_mytransform2:**
    
    | Biến | GVIF | Df | $GVIF^{1/(2 \cdot Df)}$ |
    | --- | --- | --- | --- |
    | Year | 1.292286 | 1 | 1.136787 |
    | Ratings | 1.316642 | 1 | 1.147450 |
    | Genre | 3.775440 | 8 | 1.086577 |
    | I(Budget^(1/5)) | 3.033832 | 1 | 1.741790 |
    | Screens | 2.577423 | 1 | 1.605436 |
    | Sequel | 2.415095 | 6 | 1.076245 |
    | I(Views^(1/4)) | 10.074697 | 1 | 3.174066 |
    | I(Likes^(1/5)) | 10.900616 | 1 | 3.301608 |
    | I(AggregateFollowers^(1/5)) | 1.497522 | 1 | 1.223733 |
    
    **4. Đối với mô hình full_model_BCtransform2:**
    
    | Biến | GVIF | Df | $GVIF^{1/(2 \cdot Df)}$ |
    | --- | --- | --- | --- |
    | Year | 1.286247 | 1 | 1.134128 |
    | Ratings | 1.313631 | 1 | 1.146137 |
    | Genre | 3.785916 | 8 | 1.086765 |
    | I(Budget^(0.2)) | 3.032214 | 1 | 1.741325 |
    | Screens | 2.578424 | 1 | 1.605747 |
    | Sequel | 2.405694 | 6 | 1.075895 |
    | I(Views^(0.3)) | 9.209687 | 1 | 3.034747 |
    | I(Likes^(0.2)) | 10.045659 | 1 | 3.169489 |
    | I(AggregateFollowers^(0.2)) | 1.496466 | 1 | 1.223301 |
    
    **Nhận xét chung:** Ta thấy hệ số phóng đại **VIF** của tất cả các biến trong cả 4 mô hình đều $\le 3.xyz$ nên cả 4 mô hình đều không có hiện tượng đa cộng tuyến.
    
    ### Kiểm định ý nghĩa của mô hình và Kiểm định ý nghĩa của từng biến đối với mô hình:
    
    *Trích xuất kết quả `summary()` cho các mô hình full_model:*
    
    **1. Mô hình full_model_mytransform:**
    
    ```
    ## Call:
    ## lm(formula = I(Gross^(1/3)) ~ Year + Ratings + Genre + I(Budget^(1/5)) +
    ##     I(Screens^(3/2)) + Sequel + I(Views^(1/4)) + I(Likes^(1/5)) +
    ##     I(AggregateFollowers^(1/5)), data = train)
    ...
    ## Residual standard error: 93.98 on 160 degrees of freedom
    ## Multiple R-squared:  0.7419,	Adjusted R-squared:  0.7081
    ## F-statistic: 21.91 on 21 and 160 DF,  p-value: < 2.2e-16
    ```
    
    **2. Mô hình full_model_BCtransform:**
    
    ```
    ## Call:
    ## lm(formula = I(Gross^(0.3)) ~ Year + Ratings + Genre + I(Budget^(0.2)) +
    ##     I(Screens^(0.5)) + Sequel + I(Views^(0.3)) + I(Likes^(0.2)) +
    ##     I(AggregateFollowers^(0.2)), data = train)
    ...
    ## Residual standard error: 48.41 on 160 degrees of freedom
    ## Multiple R-squared:  0.7371,	Adjusted R-squared:  0.7026
    ## F-statistic: 21.36 on 21 and 160 DF,  p-value: < 2.2e-16
    ```
    
    **3. Mô hình full_model_mytransform2:**
    
    ```
    ## Call:
    ## lm(formula = I(Gross^(1/3)) ~ Year + Ratings + Genre + I(Budget^(1/5)) +
    ##     Screens + Sequel + I(Views^(1/4)) + I(Likes^(1/5)) + I(AggregateFollowers^(1/5)),
    ##     data = train)
    ...
    ## Residual standard error: 48.36 on 160 degrees of freedom
    ## Multiple R-squared:  0.7377,	Adjusted R-squared:  0.7033
    ## F-statistic: 21.43 on 21 and 160 DF,  p-value: < 2.2e-16
    ```
    
    *Sinh viên cần đưa ra nhận xét cụ thể cho từng mô hình để có cái nhìn đầy đủ.*
    
    ### Thực hiện stepwise backward với tiêu chuẩn AIC để chọn ra mô hình tốt hơn/ ít biến giải thích hơn cho từng mô hình biến đổi hiện tại:
    
    *Trích xuất quá trình loại bỏ biến (ví dụ minh họa):*
    
    ```
    ## Start:  AIC=1674.25
    ## I(Gross^(1/3)) ~ Year + Ratings + Genre + I(Budget^(1/5)) + I(Screens^(3/2)) +
    ##     Sequel + I(Views^(1/4)) + I(Likes^(1/5)) + I(AggregateFollowers^(1/5))
    ...
    ## Step:  AIC=1666.72
    ## I(Gross^(1/3)) ~ Year + Ratings + I(Budget^(1/5)) + I(Screens^(3/2)) +
    ##     Sequel + I(Views^(1/4)) + I(Likes^(1/5)) + I(AggregateFollowers^(1/5))
    ...
    ```
    
    Sau khi thực hiện stepwise, ta thu được 4 mô hình sau đây (mỗi mô hình đều có 7 biến giải thích):
    
    - `short_model_mytransform`
    - `short_model_BCtransform`
    - `short_model_mytransform2`
    - `short_model_BCtransform2`
    
    ### Kiểm định ý nghĩa của mô hình và Kiểm định ý nghĩa của từng biến đối với mô hình:
    
    *Trích xuất kết quả `summary()` cho các mô hình short_model:*
    
    **1. Mô hình short_model_mytransform:**
    
    ```
    ## Call:
    ## lm(formula = I(Gross^(1/3)) ~ Year + Ratings + I(Budget^(1/5)) +
    ##     I(Screens^(3/2)) + I(Views^(1/4)) + I(Likes^(1/5)) + I(AggregateFollowers^(1/5)),
    ##     data = train)
    ...
    ```
    
    **2. Mô hình short_model_BCtransform:**
    
    ```
    ## Call:
    ## lm(formula = I(Gross^(0.3)) ~ Year + Ratings + I(Budget^(0.2)) +
    ##     I(Screens^(0.5)) + Sequel + I(Views^(0.3)) + I(Likes^(0.2)),
    ##     data = train)
    ...
    ## Residual standard error: 48.52 on 169 degrees of freedom
    ## Multiple R-squared:  0.7211,	Adjusted R-squared:  0.7013
    ## F-statistic: 36.41 on 12 and 169 DF,  p-value: < 2.2e-16
    ```
    
    ### So sánh $R^2_{adj}$ của các mô hình:
    
    ```
    ## R^2 hiệu chỉnh của short_model_mytransform = 0.7021036
    ```
    
    ```
    ## R^2 hiệu chỉnh của short_model_BCtransform = 0.7012744
    ```
    
    ```
    ## R^2 hiệu chỉnh của short_model_mytransform2 = 0.7094016
    ```
    
    ```
    ## R^2 hiệu chỉnh của short_model_BCtransform2 = 0.7038545
    ```
    
    Từ kết quả trên, ta thấy các mô hình rút gọn đều có mức độ giải thích $R^2_{adj}$ xung quanh 70% - 71%. Trong đó, mô hình `short_model_mytransform2` cho kết quả $R^2_{adj}$ tốt nhất (~0.709).
    
    ### Kiểm tra giả định hồi quy tuyến tính:
    
    - **Kiểm tra bằng thống kê mô tả:**
        
        ![image.png](image%2023.png)
        
        ![image.png](image%2024.png)
        
        ![image.png](image%2025.png)
        
        ![image.png](image%2026.png)
        
        *Sinh viên cần đưa ra nhận xét cụ thể.*
        
        - **Kiểm tra bằng thống kê suy diễn:**
        
        *(Kết quả kiểm định cho các mô hình rút gọn)*
        
        ```
        ## p-value = 0.2559506 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: sai số của mô hình tuân theo phân phối chuẩn.
        ```
        
        ```
        ## p-value = 1 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: E(sai số) = 0 với khả năng sai lầm 5%.
        ```
        
        ```
        ## p-value = 0.3960459 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: mô hình có phương sai ổn định.
        ```
        
        ```
        ## p-value = 0.07015809 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: sai số của mô hình tuân theo phân phối chuẩn.
        ```
        
        ```
        ## p-value = 1 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: E(sai số) = 0 với khả năng sai lầm 5%.
        ```
        
        ```
        ## p-value = 0.4486462 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: mô hình có phương sai ổn định.
        ```
        
        ```
        ## p-value = 0.0904547 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: sai số của mô hình tuân theo phân phối chuẩn.
        ```
        
        ```
        ## p-value = 1 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: E(sai số) = 0 với khả năng sai lầm 5%.
        ```
        
        ```
        ## p-value = 0.6684323 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: mô hình có phương sai ổn định.
        ```
        
        ```
        ## p-value = 0.07678846 nên ta chấp nhận H0 với khả năng sai lầm 5 %. Hay nói cách khác: sai số của mô hình tuân theo phân phối chuẩn.
        ```
        
        **Nhận xét:** cả 4 mô hình biến đổi biến thu được sau stepwise đều thỏa mãn giả định hồi quy tuyến tính.
        

## 5. Dự báo (Prediction)

Ta tiến hành dự báo doanh thu phim dựa trên các mô hình tìm được và trên bộ dữ liệu Validation:

```
##   Gross_original Gross_full_model Gross_short_model Gross_hat_mytransform
## 1       2.90e+04        -42128675         -39539397               10356.3
## 2       4.26e+07         70205791          84019560            58233416.4
## 3       5.75e+06         27730029          28425676            11225823.2
## 4       2.60e+07         39765949          50330960            30434000.6
## 5       3.50e+08         93942571          99544899           101755106.9
## 6       1.52e+07         17022239          24901277            11876985.1
##   Gross_hat_BCtransform Gross_hat_mytransform2 Gross_hat_BCtransform2
## 1              5573.091           4.336959e+02               12821.88
## 2          54898043.277           5.875331e+07            56756964.35
## 3          17932886.460           1.325476e+07            12869112.48
## 4          36477060.687           3.339095e+07            35103542.50
## 5          92473751.872           9.814831e+07            98977130.32
## 6          17870550.063           1.470422e+07            13934641.07
##   Gross_hat_short_mytransform Gross_hat_short_BCtransform
## 1                    48382.17                    21107.61
## 2                 79760156.28                 61169796.85
## 3                 10287055.51                 18768338.95
## 4                 31041707.25                 38117987.24
## 5                110391988.13                104502061.67
## 6                 17240560.25                 23273301.56
```

Ta thấy có tất cả 42 giá trị dự báo cho mỗi mô hình (tương ứng 42 quan trắc trong bộ Validation).

Kiểm tra số lượng giá trị NA trong tập kết quả:

```
##               Gross_original             Gross_full_model
##                            0                            1
##            Gross_short_model        Gross_hat_mytransform
##                            0                            1
##        Gross_hat_BCtransform       Gross_hat_mytransform2
##                            1                            1
##       Gross_hat_BCtransform2  Gross_hat_short_mytransform
##                            1                            0
##  Gross_hat_short_BCtransform Gross_hat_short_mytransform2
##                            1                            0
## Gross_hat_short_BCtransform2
##                            0
```

Do có 1 dòng dữ liệu chứa giá trị NA (ở một số mô hình), ta tiến hành loại bỏ dòng quan trắc này. Kết quả kiểm tra lại lượng NA:

```
##               Gross_original             Gross_full_model
##                            0                            0
##            Gross_short_model        Gross_hat_mytransform
##                            0                            0
##        Gross_hat_BCtransform       Gross_hat_mytransform2
##                            0                            0
##       Gross_hat_BCtransform2  Gross_hat_short_mytransform
##                            0                            0
##  Gross_hat_short_BCtransform Gross_hat_short_mytransform2
##                            0                            0
## Gross_hat_short_BCtransform2
##                            0
```

Bảng dữ liệu kết quả sau khi đã xóa NA (hiển thị 6 dòng đầu):

```
##   Gross_original Gross_full_model Gross_short_model Gross_hat_mytransform
## 1       2.90e+04        -42128675         -39539397               10356.3
## 2       4.26e+07         70205791          84019560            58233416.4
## 3       5.75e+06         27730029          28425676            11225823.2
## 4       2.60e+07         39765949          50330960            30434000.6
## 5       3.50e+08         93942571          99544899           101755106.9
## 6       1.52e+07         17022239          24901277            11876985.1
##   Gross_hat_BCtransform Gross_hat_mytransform2 Gross_hat_BCtransform2
## 1              5573.091           4.336959e+02               12821.88
## 2          54898043.277           5.875331e+07            56756964.35
## 3          17932886.460           1.325476e+07            12869112.48
## 4          36477060.687           3.339095e+07            35103542.50
## 5          92473751.872           9.814831e+07            98977130.32
## 6          17870550.063           1.470422e+07            13934641.07
##   Gross_hat_short_mytransform Gross_hat_short_BCtransform
## 1                    48382.17                    21107.61
## 2                 79760156.28                 61169796.85
## 3                 10287055.51                 18768338.95
## 4                 31041707.25                 38117987.24
## 5                110391988.13                104502061.67
## 6                 17348569.25                 23272201.56
##   Gross_hat_short_mytransform2 Gross_hat_short_BCtransform2
## 1                 6.940548e+03                     37375.44
## 2                 6.472678e+07                  63812636.32
## 3                 1.363774e+07                  13157748.62
## 4                 3.551937e+07                  35172275.78
## 5                 1.114935e+08                 111483866.92
## 6                 1.926861e+07                  18357443.41
```

```
## [1] 41 11
```

Sau khi xóa kết quả NA, ta còn 41 giá trị dự báo cho mỗi mô hình.

## 6. Đánh giá mô hình (Evaluation)

![image.png](image%2027.png)

- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error