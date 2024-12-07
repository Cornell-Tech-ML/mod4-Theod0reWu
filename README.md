# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

<br>

## Sentiment analysis

Settings changed:

- Training Size: 1000
- Batch size: 50
- Learning Rate: 1e-8
- Max Epochs: 40

```
missing pre-trained embedding for 108 unknown words
Epoch 1, loss 691.6816884072938, train accuracy: 53.70%
Validation accuracy: 55.00%
Best Valid accuracy: 55.00%
Epoch 2, loss 691.7633799198682, train accuracy: 53.10%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 3, loss 691.9739474941315, train accuracy: 53.40%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 4, loss 691.8427753801449, train accuracy: 52.80%
Validation accuracy: 60.00%
Best Valid accuracy: 73.00%
Epoch 5, loss 691.857505772609, train accuracy: 53.90%
Validation accuracy: 56.00%
Best Valid accuracy: 73.00%
Epoch 6, loss 691.8793807942442, train accuracy: 53.10%
Validation accuracy: 53.00%
Best Valid accuracy: 73.00%
Epoch 7, loss 691.922742737525, train accuracy: 51.40%
Validation accuracy: 63.00%
Best Valid accuracy: 73.00%
Epoch 8, loss 691.9809309074121, train accuracy: 54.10%
Validation accuracy: 56.00%
Best Valid accuracy: 73.00%
Epoch 9, loss 691.8360356464489, train accuracy: 51.30%
Validation accuracy: 56.00%
Best Valid accuracy: 73.00%
Epoch 10, loss 691.8320865041296, train accuracy: 53.00%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 11, loss 691.921917266305, train accuracy: 51.30%
Validation accuracy: 57.00%
Best Valid accuracy: 73.00%
Epoch 12, loss 691.799953204906, train accuracy: 54.70%
Validation accuracy: 57.00%
Best Valid accuracy: 73.00%
Epoch 13, loss 691.8493718168545, train accuracy: 52.80%
Validation accuracy: 57.00%
Best Valid accuracy: 73.00%
Epoch 14, loss 691.9886079541163, train accuracy: 52.20%
Validation accuracy: 55.00%
Best Valid accuracy: 73.00%
Epoch 15, loss 691.9069279480101, train accuracy: 53.10%
Validation accuracy: 60.00%
Best Valid accuracy: 73.00%
Epoch 16, loss 691.8733938291488, train accuracy: 52.40%
Validation accuracy: 62.00%
Best Valid accuracy: 73.00%
Epoch 17, loss 691.8082806520836, train accuracy: 50.80%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 18, loss 691.8493068018454, train accuracy: 54.10%
Validation accuracy: 55.00%
Best Valid accuracy: 73.00%
Epoch 19, loss 691.8277664955407, train accuracy: 51.90%
Validation accuracy: 61.00%
Best Valid accuracy: 73.00%
Epoch 20, loss 692.008466830097, train accuracy: 54.50%
Validation accuracy: 64.00%
Best Valid accuracy: 73.00%
Epoch 21, loss 692.0646078153474, train accuracy: 51.10%
Validation accuracy: 56.00%
Best Valid accuracy: 73.00%
Epoch 22, loss 691.8459180790242, train accuracy: 53.60%
Validation accuracy: 66.00%
Best Valid accuracy: 73.00%
Epoch 23, loss 691.8253524147333, train accuracy: 51.50%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 24, loss 691.933168332068, train accuracy: 53.60%
Validation accuracy: 57.00%
Best Valid accuracy: 73.00%
Epoch 25, loss 691.9108666841918, train accuracy: 53.60%
Validation accuracy: 50.00%
Best Valid accuracy: 73.00%
Epoch 26, loss 691.8364168871708, train accuracy: 53.80%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 27, loss 691.8390262908727, train accuracy: 53.30%
Validation accuracy: 61.00%
Best Valid accuracy: 73.00%
Epoch 28, loss 691.8557871118712, train accuracy: 52.30%
Validation accuracy: 61.00%
Best Valid accuracy: 73.00%
Epoch 29, loss 691.8054054780387, train accuracy: 51.90%
Validation accuracy: 57.00%
Best Valid accuracy: 73.00%
Epoch 30, loss 691.8419445136428, train accuracy: 53.60%
Validation accuracy: 69.00%
Best Valid accuracy: 73.00%
Epoch 31, loss 691.9125221011918, train accuracy: 51.90%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 32, loss 691.9096407498974, train accuracy: 51.20%
Validation accuracy: 65.00%
Best Valid accuracy: 73.00%
Epoch 33, loss 691.7979448999379, train accuracy: 53.20%
Validation accuracy: 60.00%
Best Valid accuracy: 73.00%
Epoch 34, loss 692.0106141216281, train accuracy: 51.50%
Validation accuracy: 60.00%
Best Valid accuracy: 73.00%
Epoch 35, loss 691.8505455164418, train accuracy: 53.00%
Validation accuracy: 62.00%
Best Valid accuracy: 73.00%
Epoch 36, loss 691.8087167172374, train accuracy: 52.50%
Validation accuracy: 61.00%
Best Valid accuracy: 73.00%
Epoch 37, loss 691.8823905488551, train accuracy: 52.00%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 38, loss 691.9159864792301, train accuracy: 51.70%
Validation accuracy: 60.00%
Best Valid accuracy: 73.00%
Epoch 39, loss 691.8754890288681, train accuracy: 52.40%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
Epoch 40, loss 691.8573913201045, train accuracy: 52.20%
Validation accuracy: 58.00%
Best Valid accuracy: 73.00%
```

## MNIST dataset

```
Epoch 1 loss 2.302225771644656 valid acc 3/16
Epoch 1 loss 11.515899483639192 valid acc 1/16
Epoch 1 loss 11.481670268997265 valid acc 5/16
Epoch 1 loss 11.425595086570304 valid acc 5/16
Epoch 1 loss 11.249528046311237 valid acc 5/16
Epoch 1 loss 11.027098194630248 valid acc 6/16
Epoch 1 loss 10.234282294781778 valid acc 8/16
Epoch 1 loss 9.30757888052126 valid acc 10/16
Epoch 1 loss 9.589529997825288 valid acc 9/16
Epoch 1 loss 9.824923821054112 valid acc 9/16
Epoch 1 loss 8.360372925523727 valid acc 9/16
Epoch 1 loss 8.731068510487448 valid acc 8/16
Epoch 1 loss 7.7972297760106235 valid acc 10/16
Epoch 1 loss 6.954778714861846 valid acc 9/16
Epoch 1 loss 6.92336425587721 valid acc 10/16
Epoch 1 loss 6.3287708550244295 valid acc 8/16
Epoch 1 loss 7.483072360526515 valid acc 10/16
Epoch 1 loss 6.301888506673193 valid acc 9/16
Epoch 1 loss 6.5719370807210655 valid acc 12/16
Epoch 1 loss 6.7132716244409725 valid acc 11/16
Epoch 1 loss 5.002512247750267 valid acc 11/16
Epoch 1 loss 4.89128347474673 valid acc 14/16
Epoch 1 loss 3.4150335403077845 valid acc 11/16
Epoch 1 loss 4.638158683662136 valid acc 12/16
Epoch 1 loss 4.371426670653963 valid acc 10/16
Epoch 1 loss 5.1472105789710305 valid acc 8/16
Epoch 1 loss 6.432346953579482 valid acc 12/16
Epoch 1 loss 3.8853997100407933 valid acc 13/16
Epoch 1 loss 3.7819618478129713 valid acc 13/16
Epoch 1 loss 2.747100219104823 valid acc 12/16
Epoch 1 loss 5.781349568220445 valid acc 12/16
Epoch 1 loss 5.26125470881346 valid acc 11/16
Epoch 1 loss 3.80064174880157 valid acc 11/16
Epoch 1 loss 4.83999071739886 valid acc 10/16
Epoch 1 loss 7.734888394485319 valid acc 10/16
Epoch 1 loss 4.5621305044853475 valid acc 14/16
Epoch 1 loss 3.8952620027793916 valid acc 11/16
Epoch 1 loss 4.534296650262571 valid acc 12/16
Epoch 1 loss 3.989235705533749 valid acc 11/16
Epoch 1 loss 4.693826236512848 valid acc 10/16
Epoch 1 loss 4.1039610336840955 valid acc 14/16
Epoch 1 loss 3.7321883231210093 valid acc 15/16
Epoch 1 loss 3.3989534723088 valid acc 12/16
Epoch 1 loss 3.1874265548176943 valid acc 13/16
Epoch 1 loss 4.416314536983306 valid acc 14/16
Epoch 1 loss 2.7730345268482974 valid acc 13/16
Epoch 1 loss 3.8637732790049792 valid acc 13/16
Epoch 1 loss 3.53598654006522 valid acc 14/16
Epoch 1 loss 2.9786516293426537 valid acc 13/16
Epoch 1 loss 3.176330060872448 valid acc 15/16
Epoch 1 loss 3.395344305264995 valid acc 14/16
Epoch 1 loss 3.1272135513944805 valid acc 15/16
Epoch 1 loss 3.4175412031116243 valid acc 13/16
Epoch 1 loss 2.3974694064769513 valid acc 13/16
Epoch 1 loss 4.011037617056332 valid acc 15/16
Epoch 1 loss 2.2993770608464357 valid acc 12/16
Epoch 1 loss 2.90747009810167 valid acc 14/16
Epoch 1 loss 2.4803299181846397 valid acc 12/16
Epoch 1 loss 3.163591472474093 valid acc 14/16
Epoch 1 loss 4.112557464063548 valid acc 14/16
Epoch 1 loss 3.3565961830627744 valid acc 14/16
Epoch 1 loss 3.6305885561487092 valid acc 13/16
Epoch 1 loss 3.240365335211033 valid acc 13/16
Epoch 2 loss 0.15517306584267815 valid acc 15/16
Epoch 2 loss 2.097982050837629 valid acc 15/16
Epoch 2 loss 3.7508247695545003 valid acc 14/16
Epoch 2 loss 2.4742914440632857 valid acc 15/16
Epoch 2 loss 2.7071547154947595 valid acc 15/16
Epoch 2 loss 2.01244032323395 valid acc 14/16
Epoch 2 loss 2.495597688374223 valid acc 14/16
Epoch 2 loss 3.1193746286181114 valid acc 15/16
Epoch 2 loss 4.63675771260353 valid acc 13/16
Epoch 2 loss 2.298877417770689 valid acc 14/16
Epoch 2 loss 1.7962201383127745 valid acc 14/16
Epoch 2 loss 2.9076996877408083 valid acc 14/16
Epoch 2 loss 3.037060141574477 valid acc 13/16
Epoch 2 loss 3.907443878294055 valid acc 14/16
Epoch 2 loss 2.505133390054834 valid acc 12/16
Epoch 2 loss 2.14585351674325 valid acc 14/16
Epoch 2 loss 2.7197839248383398 valid acc 13/16
Epoch 2 loss 2.880370192481486 valid acc 14/16
Epoch 2 loss 2.2165661113847563 valid acc 14/16
Epoch 2 loss 1.9118003126026533 valid acc 15/16
Epoch 2 loss 1.6256534749615033 valid acc 13/16
Epoch 2 loss 1.4068571762844295 valid acc 13/16
Epoch 2 loss 0.4520364838768845 valid acc 13/16
Epoch 2 loss 1.6011342058562306 valid acc 15/16
Epoch 2 loss 1.6353103701962377 valid acc 14/16
Epoch 2 loss 2.8097949040845673 valid acc 13/16
Epoch 2 loss 2.5500525721720932 valid acc 15/16
Epoch 2 loss 1.737529405860053 valid acc 16/16
Epoch 2 loss 1.2203567704960467 valid acc 11/16
Epoch 2 loss 0.856057526115281 valid acc 14/16
Epoch 2 loss 1.9773789373503767 valid acc 15/16
Epoch 2 loss 2.692138091757122 valid acc 13/16
Epoch 2 loss 1.382904527548458 valid acc 14/16
Epoch 2 loss 1.90513320733411 valid acc 13/16
Epoch 2 loss 2.532705819646792 valid acc 13/16
Epoch 2 loss 1.7773240435085307 valid acc 12/16
Epoch 2 loss 0.9232962305859852 valid acc 13/16
Epoch 2 loss 1.4212244845863617 valid acc 15/16
Epoch 2 loss 1.3840397411855903 valid acc 13/16
Epoch 2 loss 1.6410710118727123 valid acc 15/16
Epoch 2 loss 1.1881356996568835 valid acc 13/16
Epoch 2 loss 2.051621104081021 valid acc 15/16
Epoch 2 loss 1.2680012150561957 valid acc 14/16
Epoch 2 loss 1.0716829672182884 valid acc 13/16
Epoch 2 loss 1.9716269417563228 valid acc 15/16
Epoch 2 loss 1.18945427682639 valid acc 13/16
Epoch 2 loss 1.6713792327977117 valid acc 15/16
Epoch 2 loss 1.9897758883873795 valid acc 14/16
Epoch 2 loss 0.9862271955215474 valid acc 14/16
Epoch 2 loss 1.5260065231044966 valid acc 16/16
Epoch 2 loss 2.0253237156739465 valid acc 14/16
Epoch 2 loss 1.5554745698764083 valid acc 13/16
Epoch 2 loss 2.296891089088299 valid acc 14/16
Epoch 2 loss 1.1606610428461313 valid acc 14/16
Epoch 2 loss 1.8894874282551664 valid acc 14/16
Epoch 2 loss 1.1817011560325124 valid acc 14/16
Epoch 2 loss 1.1746917408378974 valid acc 15/16
Epoch 2 loss 1.4487042846261764 valid acc 14/16
Epoch 2 loss 1.7034882041589814 valid acc 13/16
Epoch 2 loss 1.775604803866914 valid acc 14/16
Epoch 2 loss 1.4349404630710156 valid acc 13/16
Epoch 2 loss 1.2749639780849673 valid acc 15/16
Epoch 2 loss 1.3093381981143999 valid acc 15/16
Epoch 3 loss 0.302297183289405 valid acc 15/16
Epoch 3 loss 2.12774802424173 valid acc 14/16
Epoch 3 loss 1.9214739181671017 valid acc 14/16
Epoch 3 loss 0.9692284940662945 valid acc 15/16
Epoch 3 loss 1.3075583822766719 valid acc 14/16
Epoch 3 loss 1.5507019182625679 valid acc 15/16
Epoch 3 loss 1.9007577847193788 valid acc 12/16
Epoch 3 loss 1.4680796549608979 valid acc 15/16
Epoch 3 loss 1.8418080025139851 valid acc 15/16
Epoch 3 loss 0.6524196309510857 valid acc 15/16
Epoch 3 loss 0.9983860288907713 valid acc 16/16
Epoch 3 loss 1.7851174715137261 valid acc 13/16
Epoch 3 loss 2.350199761052321 valid acc 14/16
Epoch 3 loss 1.7579267785815174 valid acc 15/16
Epoch 3 loss 1.6970409101258834 valid acc 14/16
Epoch 3 loss 1.3872047272621242 valid acc 14/16
Epoch 3 loss 1.8338898325591522 valid acc 14/16
Epoch 3 loss 2.076721693215296 valid acc 14/16
Epoch 3 loss 2.1748828061830174 valid acc 14/16
Epoch 3 loss 0.9099799655465491 valid acc 15/16
Epoch 3 loss 1.1758398494417002 valid acc 14/16
Epoch 3 loss 0.9288464960684227 valid acc 13/16
Epoch 3 loss 0.5296999846365663 valid acc 13/16
Epoch 3 loss 0.6164663080781542 valid acc 15/16
Epoch 3 loss 1.498739484967619 valid acc 14/16
Epoch 3 loss 1.5494135171483046 valid acc 16/16
Epoch 3 loss 0.8969153696047352 valid acc 15/16
Epoch 3 loss 0.9377781939252875 valid acc 15/16
Epoch 3 loss 0.9141936987245824 valid acc 15/16
Epoch 3 loss 0.7101376269756714 valid acc 16/16
Epoch 3 loss 1.289535395110764 valid acc 16/16
Epoch 3 loss 1.2172283912997464 valid acc 14/16
Epoch 3 loss 0.9399307310590882 valid acc 13/16
Epoch 3 loss 1.5129577315586196 valid acc 14/16
Epoch 3 loss 1.7043218805884361 valid acc 14/16
Epoch 3 loss 0.9845637087092012 valid acc 14/16
Epoch 3 loss 0.7104110660066898 valid acc 15/16
Epoch 3 loss 0.8692479997564312 valid acc 15/16
Epoch 3 loss 0.9725002208329366 valid acc 14/16
Epoch 3 loss 1.6576056842279407 valid acc 14/16
Epoch 3 loss 1.1843860795943628 valid acc 14/16
Epoch 3 loss 0.9034948070096814 valid acc 16/16
Epoch 3 loss 0.9810017124429934 valid acc 13/16
Epoch 3 loss 0.6017950792349993 valid acc 14/16
Epoch 3 loss 1.8566446681525033 valid acc 14/16
Epoch 3 loss 0.5943467477796984 valid acc 15/16
Epoch 3 loss 1.226907849954792 valid acc 15/16
Epoch 3 loss 1.8808006528959544 valid acc 14/16
Epoch 3 loss 0.8698000351639664 valid acc 15/16
Epoch 3 loss 0.983661985545989 valid acc 14/16
Epoch 3 loss 1.2600757496269785 valid acc 14/16
Epoch 3 loss 0.8344862763588349 valid acc 15/16
Epoch 3 loss 1.771799874360433 valid acc 14/16
Epoch 3 loss 0.9448067690804419 valid acc 14/16
Epoch 3 loss 1.2024886817323954 valid acc 14/16
Epoch 3 loss 0.8278217090089467 valid acc 14/16
Epoch 3 loss 0.9711183569137349 valid acc 14/16
Epoch 3 loss 1.3856690510927352 valid acc 14/16
Epoch 3 loss 1.0420793589065385 valid acc 14/16
Epoch 3 loss 0.8208825963197449 valid acc 16/16
Epoch 3 loss 1.9109595199097895 valid acc 14/16
Epoch 3 loss 1.22678205978361 valid acc 14/16
Epoch 3 loss 1.235477779473579 valid acc 15/16
Epoch 4 loss 0.1180327546631388 valid acc 14/16
Epoch 4 loss 1.1645087618497119 valid acc 15/16
Epoch 4 loss 1.0685394188247397 valid acc 16/16
Epoch 4 loss 1.0680658967833652 valid acc 14/16
Epoch 4 loss 0.517936642769035 valid acc 14/16
Epoch 4 loss 0.7205496346906145 valid acc 16/16
Epoch 4 loss 1.5918547714493942 valid acc 16/16
Epoch 4 loss 1.239129657796794 valid acc 16/16
Epoch 4 loss 1.0282205260679311 valid acc 15/16
Epoch 4 loss 0.6800909101659963 valid acc 16/16
Epoch 4 loss 0.7420215373250314 valid acc 16/16
Epoch 4 loss 2.2914509046394986 valid acc 16/16
Epoch 4 loss 1.553731907270529 valid acc 14/16
Epoch 4 loss 1.4417502342013706 valid acc 14/16
Epoch 4 loss 1.376767316244387 valid acc 15/16
Epoch 4 loss 0.9115136568054847 valid acc 16/16
Epoch 4 loss 1.583891116440337 valid acc 15/16
Epoch 4 loss 1.4891463215000549 valid acc 15/16
Epoch 4 loss 1.44952324470052 valid acc 14/16
Epoch 4 loss 0.791530678578429 valid acc 16/16
Epoch 4 loss 1.2427307581182585 valid acc 12/16
Epoch 4 loss 0.7115271856856058 valid acc 14/16
Epoch 4 loss 0.38002654624125676 valid acc 15/16
Epoch 4 loss 1.1496148043103367 valid acc 15/16
Epoch 4 loss 1.244245656915774 valid acc 15/16
Epoch 4 loss 2.388934784894951 valid acc 14/16
Epoch 4 loss 1.6588606799549095 valid acc 15/16
Epoch 4 loss 1.394520468506774 valid acc 16/16
Epoch 4 loss 0.85173890697169 valid acc 15/16
Epoch 4 loss 0.31295343941262865 valid acc 16/16
Epoch 4 loss 0.9333371382552076 valid acc 13/16
Epoch 4 loss 1.1221139237230435 valid acc 15/16
Epoch 4 loss 1.1211814365286084 valid acc 13/16
Epoch 4 loss 0.6691135542014068 valid acc 15/16
Epoch 4 loss 1.0510689554603805 valid acc 14/16
Epoch 4 loss 1.343227219806553 valid acc 15/16
Epoch 4 loss 1.1838942680084157 valid acc 13/16
Epoch 4 loss 1.1494652276947712 valid acc 14/16
Epoch 4 loss 1.021694515642244 valid acc 15/16
Epoch 4 loss 0.6043540707749858 valid acc 15/16
Epoch 4 loss 0.9666970335970236 valid acc 14/16
Epoch 4 loss 0.947131342725287 valid acc 15/16
Epoch 4 loss 0.6328738489590737 valid acc 15/16
Epoch 4 loss 0.14477814641102063 valid acc 14/16
Epoch 4 loss 1.3084668188677577 valid acc 15/16
Epoch 4 loss 0.4964892090653914 valid acc 15/16
Epoch 4 loss 0.8561759078070533 valid acc 16/16
Epoch 4 loss 1.2986979265809495 valid acc 15/16
Epoch 4 loss 0.8292871525040553 valid acc 13/16
Epoch 4 loss 0.9597639976017208 valid acc 15/16
Epoch 4 loss 1.0633234077993454 valid acc 16/16
Epoch 4 loss 0.8871756395420977 valid acc 16/16
Epoch 4 loss 0.8772290123938475 valid acc 14/16
Epoch 4 loss 0.5157603063347598 valid acc 15/16
Epoch 4 loss 1.130976179715413 valid acc 14/16
Epoch 4 loss 0.6902881825740673 valid acc 14/16
Epoch 4 loss 0.5426775753117885 valid acc 16/16
Epoch 4 loss 0.8622628392250047 valid acc 16/16
Epoch 4 loss 0.8572541058070289 valid acc 15/16
Epoch 4 loss 0.8134898056324192 valid acc 15/16
Epoch 4 loss 1.2307960085483278 valid acc 14/16
Epoch 4 loss 0.7009993865681886 valid acc 15/16
Epoch 4 loss 0.8138530595541296 valid acc 16/16
Epoch 5 loss 0.005417706290827951 valid acc 15/16
Epoch 5 loss 1.0069947354607152 valid acc 15/16
Epoch 5 loss 1.5047136015662324 valid acc 16/16
Epoch 5 loss 1.1339205845146418 valid acc 15/16
Epoch 5 loss 0.3247031457714447 valid acc 16/16
Epoch 5 loss 0.4380091311119601 valid acc 16/16
Epoch 5 loss 1.0733986563743367 valid acc 16/16
Epoch 5 loss 0.9811846409184691 valid acc 16/16
Epoch 5 loss 0.6698997947221415 valid acc 15/16
Epoch 5 loss 0.4534408833513203 valid acc 13/16
Epoch 5 loss 0.6301702294072091 valid acc 15/16
Epoch 5 loss 1.436833427974266 valid acc 14/16
Epoch 5 loss 1.8680107103498775 valid acc 16/16
Epoch 5 loss 1.1211505893074594 valid acc 16/16
Epoch 5 loss 1.3204295094805185 valid acc 16/16
Epoch 5 loss 0.7220837541008158 valid acc 16/16
Epoch 5 loss 1.6600158582422444 valid acc 13/16
Epoch 5 loss 1.0272629538084828 valid acc 16/16
Epoch 5 loss 1.0075509708173385 valid acc 14/16
Epoch 5 loss 0.6300225404934752 valid acc 15/16
Epoch 5 loss 1.064529591334824 valid acc 15/16
Epoch 5 loss 0.9302406434373911 valid acc 14/16
Epoch 5 loss 0.3427724268783131 valid acc 14/16
Epoch 5 loss 0.4064703426734293 valid acc 15/16
Epoch 5 loss 0.6732432509469147 valid acc 13/16
Epoch 5 loss 1.9502455084087134 valid acc 15/16
Epoch 5 loss 0.7225020138262818 valid acc 16/16
Epoch 5 loss 0.90448752259219 valid acc 15/16
Epoch 5 loss 0.39257305491846944 valid acc 14/16
Epoch 5 loss 0.14437510501698336 valid acc 15/16
Epoch 5 loss 1.0047150387346275 valid acc 16/16
Epoch 5 loss 0.37807287247950055 valid acc 16/16
Epoch 5 loss 0.627215928025935 valid acc 15/16
Epoch 5 loss 0.7256108302934301 valid acc 13/16
Epoch 5 loss 1.1960147711277465 valid acc 14/16
Epoch 5 loss 0.5026736773161133 valid acc 15/16
Epoch 5 loss 0.37373109608653826 valid acc 15/16
Epoch 5 loss 0.5072255237596895 valid acc 16/16
Epoch 5 loss 1.0011186469574185 valid acc 15/16
Epoch 5 loss 0.786042023857611 valid acc 16/16
Epoch 5 loss 0.37075199848232904 valid acc 15/16
Epoch 5 loss 0.3252867658214547 valid acc 16/16
Epoch 5 loss 0.8111412728712659 valid acc 16/16
Epoch 5 loss 0.3852300710251672 valid acc 16/16
Epoch 5 loss 1.1969051808371198 valid acc 15/16
Epoch 5 loss 0.5316959143653469 valid acc 16/16
Epoch 5 loss 0.8770552821156778 valid acc 16/16
Epoch 5 loss 2.0978766293381153 valid acc 14/16
Epoch 5 loss 0.705133014508554 valid acc 14/16
Epoch 5 loss 0.6702950389254874 valid acc 15/16
Epoch 5 loss 0.9948037030557599 valid acc 15/16
Epoch 5 loss 0.6276767145796556 valid acc 15/16
Epoch 5 loss 1.2508889299481205 valid acc 14/16
Epoch 5 loss 0.4321741104426864 valid acc 16/16
Epoch 5 loss 0.9863655833060716 valid acc 15/16
Epoch 5 loss 0.3826382059879971 valid acc 15/16
Epoch 5 loss 0.5523599027530118 valid acc 15/16
Epoch 5 loss 0.9786770615681532 valid acc 16/16
Epoch 5 loss 1.0502317348353094 valid acc 14/16
Epoch 5 loss 1.2293018864068592 valid acc 16/16
Epoch 5 loss 0.4613861474292823 valid acc 16/16
Epoch 5 loss 0.3294258578999729 valid acc 16/16
Epoch 5 loss 0.6177483764118886 valid acc 16/16
Epoch 6 loss 0.09768160934683429 valid acc 16/16
Epoch 6 loss 0.9229138729888473 valid acc 16/16
Epoch 6 loss 1.041376239074795 valid acc 15/16
Epoch 6 loss 1.173195229227435 valid acc 16/16
```
