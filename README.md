# weka.core.study

# Simple examples
- Classification
- Regression
- Clustering

```bash
101 instances loaded.
Selected attributes: 12,3,7,2,0,1,8,9,13,4,11,5,15,10,6,14,16
J48 unpruned tree
------------------

feathers = false
|   milk = false
|   |   backbone = false
|   |   |   airborne = false
|   |   |   |   predator = false
|   |   |   |   |   legs <= 2: invertebrate (2.0)
|   |   |   |   |   legs > 2: insect (2.0)
|   |   |   |   predator = true: invertebrate (8.0)
|   |   |   airborne = true: insect (6.0)
|   |   backbone = true
|   |   |   fins = false
|   |   |   |   tail = false: amphibian (3.0)
|   |   |   |   tail = true: reptile (6.0/1.0)
|   |   |   fins = true: fish (13.0)
|   milk = true: mammal (41.0)
feathers = true: bird (20.0)

Number of Leaves  : 	9

Size of the tree : 	17

mammal

Correctly Classified Instances          93               92.0792 %
Incorrectly Classified Instances         8                7.9208 %
Kappa statistic                          0.8955
Mean absolute error                      0.0225
Root mean squared error                  0.14  
Relative absolute error                 10.2478 %
Root relative squared error             42.4398 %
Total Number of Instances              101     

=== Confusion Matrix ===

  a  b  c  d  e  f  g   <-- classified as
 41  0  0  0  0  0  0 |  a = mammal
  0 20  0  0  0  0  0 |  b = bird
  0  0  3  1  0  1  0 |  c = reptile
  0  0  0 13  0  0  0 |  d = fish
  0  0  1  0  3  0  0 |  e = amphibian
  0  0  0  0  0  5  3 |  f = insect
  0  0  0  0  0  2  8 |  g = invertebrate
```

```bash
Linear Regression Model

Y1 =

    -64.774  * X1 +
     -0.0428 * X2 +
      0.0163 * X3 +
     -0.089  * X4 +
      4.1699 * X5 +
     19.9327 * X7 +
      0.2038 * X8 +
     83.9329

Correlation coefficient                  0.956 
Mean absolute error                      2.0923
Root mean squared error                  2.9569
Relative absolute error                 22.8555 %
Root relative squared error             29.282  %
Total Number of Instances              768     


M5 pruned model tree:
(using smoothed linear models)

X1 <= 0.75 : 
|   X7 <= 0.175 : 
|   |   X1 <= 0.65 : LM1 (48/1.264%)
|   |   X1 >  0.65 : LM2 (96/3.201%)
|   X7 >  0.175 : 
|   |   X1 <= 0.65 : LM3 (80/3.652%)
|   |   X1 >  0.65 : 
|   |   |   X7 <= 0.325 : LM4 (80/3.724%)
|   |   |   X7 >  0.325 : 
|   |   |   |   X1 <= 0.675 : LM5 (20/1.687%)
|   |   |   |   X1 >  0.675 : 
|   |   |   |   |   X8 <= 2.5 : LM6 (24/1.314%)
|   |   |   |   |   X8 >  2.5 : 
|   |   |   |   |   |   X8 <= 4.5 : LM7 (24/2.737%)
|   |   |   |   |   |   X8 >  4.5 : 
|   |   |   |   |   |   |   X1 <= 0.7 : LM8 (4/0.91%)
|   |   |   |   |   |   |   X1 >  0.7 : LM9 (8/1.265%)
X1 >  0.75 : 
|   X1 <= 0.805 : 
|   |   X7 <= 0.175 : LM10 (48/5.775%)
|   |   X7 >  0.175 : 
|   |   |   X7 <= 0.325 : LM11 (40/5.26%)
|   |   |   X7 >  0.325 : LM12 (40/5.756%)
|   X1 >  0.805 : 
|   |   X7 <= 0.175 : 
|   |   |   X8 <= 1.5 : 
|   |   |   |   X7 <= 0.05 : 
|   |   |   |   |   X2 <= 539 : LM13 (4/0%)
|   |   |   |   |   X2 >  539 : LM14 (12/4.501%)
|   |   |   |   X7 >  0.05 : 
|   |   |   |   |   X1 <= 0.94 : LM15 (12/4.329%)
|   |   |   |   |   X1 >  0.94 : LM16 (4/0.226%)
|   |   |   X8 >  1.5 : 
|   |   |   |   X1 <= 0.94 : LM17 (48/5.693%)
|   |   |   |   X1 >  0.94 : LM18 (16/1.119%)
|   |   X7 >  0.175 : 
|   |   |   X1 <= 0.84 : 
|   |   |   |   X7 <= 0.325 : 
|   |   |   |   |   X8 <= 2.5 : LM19 (8/3.901%)
|   |   |   |   |   X8 >  2.5 : LM20 (12/3.913%)
|   |   |   |   X7 >  0.325 : LM21 (20/5.632%)
|   |   |   X1 >  0.84 : 
|   |   |   |   X7 <= 0.325 : LM22 (60/4.548%)
|   |   |   |   X7 >  0.325 : 
|   |   |   |   |   X3 <= 306.25 : LM23 (40/4.504%)
|   |   |   |   |   X3 >  306.25 : LM24 (20/6.934%)

LM num: 1
Y1 = 
	72.2602 * X1 
	+ 0.0053 * X3 
	+ 41.5669 * X7 
	- 0.0049 * X8 
	- 37.6688

LM num: 2
Y1 = 
	-14.6772 * X1 
	+ 0.0053 * X3 
	+ 40.2316 * X7 
	+ 0.0181 * X8 
	+ 15.649

LM num: 3
Y1 = 
	84.5112 * X1 
	+ 0.0053 * X3 
	+ 13.9115 * X7 
	- 0.1471 * X8 
	- 42.4943

LM num: 4
Y1 = 
	-2.8359 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.0111 * X8 
	+ 12.0357

LM num: 5
Y1 = 
	-6.0295 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.0524 * X8 
	+ 16.0295

LM num: 6
Y1 = 
	-4.3262 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.0665 * X8 
	+ 14.5905

LM num: 7
Y1 = 
	-4.3262 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.0888 * X8 
	+ 14.5832

LM num: 8
Y1 = 
	-4.3262 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.1025 * X8 
	+ 14.5352

LM num: 9
Y1 = 
	-0.8154 * X1 
	+ 0.0053 * X3 
	+ 4.3146 * X7 
	- 0.1025 * X8 
	+ 11.9531

LM num: 10
Y1 = 
	105.9033 * X1 
	+ 0.0113 * X3 
	+ 59.6616 * X7 
	+ 0.0975 * X8 
	- 58.7462

LM num: 11
Y1 = 
	81.6537 * X1 
	+ 0.0113 * X3 
	+ 10.8932 * X7 
	+ 0.0559 * X8 
	- 33.0837

LM num: 12
Y1 = 
	64.6565 * X1 
	+ 0.0113 * X3 
	+ 10.8932 * X7 
	- 0.0337 * X8 
	- 18.0037

LM num: 13
Y1 = 
	3.2623 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	+ 44.6313 * X7 
	+ 0.0592 * X8 
	+ 11.946

LM num: 14
Y1 = 
	9.1337 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	- 0.0494 * X6 
	+ 44.6313 * X7 
	+ 0.0592 * X8 
	+ 7.321

LM num: 15
Y1 = 
	11.8776 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	- 0.0428 * X6 
	+ 44.6313 * X7 
	+ 0.0592 * X8 
	+ 7.0198

LM num: 16
Y1 = 
	3.2623 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	+ 44.6313 * X7 
	+ 0.0592 * X8 
	+ 14.1592

LM num: 17
Y1 = 
	35.1381 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	+ 16.7723 * X7 
	+ 0.0592 * X8 
	- 10.1661

LM num: 18
Y1 = 
	3.2623 * X1 
	- 0.0018 * X2 
	+ 0.0164 * X3 
	+ 16.7723 * X7 
	+ 0.0592 * X8 
	+ 16.4949

LM num: 19
Y1 = 
	8.5465 * X1 
	- 0.0012 * X2 
	+ 0.029 * X3 
	+ 15.2851 * X7 
	- 0.2151 * X8 
	+ 7.86

LM num: 20
Y1 = 
	8.5465 * X1 
	- 0.0012 * X2 
	+ 0.029 * X3 
	+ 15.2851 * X7 
	- 0.0475 * X8 
	+ 7.4789

LM num: 21
Y1 = 
	8.5465 * X1 
	- 0.0012 * X2 
	+ 0.029 * X3 
	+ 15.2851 * X7 
	+ 0.013 * X8 
	+ 8.5537

LM num: 22
Y1 = 
	1.4309 * X1 
	- 0.0012 * X2 
	+ 0.1248 * X3 
	+ 9.5464 * X7 
	+ 0.0373 * X8 
	- 10.9927

LM num: 23
Y1 = 
	5.1744 * X1 
	- 0.0012 * X2 
	+ 0.0633 * X3 
	+ 9.5464 * X7 
	+ 0.0235 * X8 
	+ 5.7355

LM num: 24
Y1 = 
	5.1744 * X1 
	- 0.0012 * X2 
	+ 0.0761 * X3 
	+ 9.5464 * X7 
	- 0.0805 * X8 
	+ 3.4386

Number of Rules : 24

Correlation coefficient                  0.9762
Mean absolute error                      1.371 
Root mean squared error                  2.1889
Relative absolute error                 14.9764 %
Root relative squared error             21.6771 %
Total Number of Instances             1536    
```

```bash
EM
==

Number of clusters selected by cross validation: 6
Number of iterations performed: 100


                 Cluster
Attribute              0        1        2        3        4        5
                   (0.1)   (0.13)   (0.26)   (0.25)   (0.12)   (0.14)
======================================================================
age
  0_34            10.0535  51.8472 122.2815  12.6207   3.1023   1.0948
  35_51           38.6282  24.4056  29.6252  89.4447  34.5208   3.3755
  52_max          13.4293    6.693   6.3459  50.8984   37.861  81.7724
  [total]         62.1111  82.9457 158.2526 152.9638  75.4841  86.2428
sex
  FEMALE          27.1812  32.2338  77.9304  83.5129  40.3199  44.8218
  MALE            33.9299  49.7119  79.3222  68.4509  34.1642   40.421
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
region
  INNER_CITY      26.1651  46.7431   73.874  60.1973  33.3759  34.6445
  TOWN            24.6991  13.0716  48.4446  53.1731   21.617  17.9946
  RURAL            8.4113  12.7871  21.7634  25.7529  11.1622  22.1231
  SUBURBAN         3.8356   11.344  15.1706  14.8404  10.3289  12.4805
  [total]         63.1111  83.9457 159.2526 153.9638  76.4841  87.2428
income
  0_24386         22.5301  77.3981 150.8728  35.3652   3.0947   1.7391
  24387_43758     38.0636   4.5119   6.2909 113.3875  70.4654   8.2808
  43759_max        1.5174   1.0357   1.0889   4.2111    1.924  76.2228
  [total]         62.1111  82.9457 158.2526 152.9638  75.4841  86.2428
married
  NO              15.0163  34.8213  48.6021  32.7954  49.5126  29.2523
  YES             46.0948  47.1244 108.6506 119.1684  24.9715  55.9904
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
children
  0                2.1776  53.2782  55.6363  92.5938   32.663  32.6511
  1               51.5497  26.7841  22.0968   1.9302  18.9418  19.6973
  2                6.4264   2.3777  56.5523  25.7573  23.3335  25.5529
  3                2.9574   1.5057  24.9671  33.6825   1.5458   9.3415
  [total]         63.1111  83.9457 159.2526 153.9638  76.4841  87.2428
car
  NO              29.7462  47.4075  89.7372  69.5918  34.7847  38.7326
  YES             31.3649  34.5382  67.5154   82.372  39.6993  46.5101
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
save_act
  NO               6.7118  58.9844  49.6095  39.7853  35.7784   1.1306
  YES             54.3993  22.9613 107.6431 112.1785  38.7056  84.1121
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
current_act
  NO              12.8656  21.8946  35.3337  46.1845  15.9243  18.7973
  YES             48.2455  60.0511 121.9189 105.7792  58.5598  66.4455
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
mortgage
  NO              34.2814  47.6791 108.1248  95.3628  54.1015  57.4504
  YES             26.8297  34.2666  49.1278   56.601  20.3826  27.7924
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
pep
  YES             59.0226  72.2592  18.5799   3.8416  68.4764  57.8202
  NO               2.0885   9.6865 138.6727 148.1222   6.0076  27.4226
  [total]         61.1111  81.9457 157.2526 151.9638  74.4841  85.2428
```