# BUILDING A SIMPLE FACE DETECTION AND TRACKING MODEL
## ABOUT DATASET
@article{Milborrow10,<br>
  author={S. Milborrow and J. Morkel and F. Nicolls},<br>
  title={{The MUCT Landmarked Face Database}},<br>
  journal={Pattern Recognition Association of South Africa},<br>
  year=2010,<br>
  note={\url{http://www.milbo.org/muct}}<br>
}
## PREPROCESS DATASET
<ol>
  <li>Using <a href="https://github.com/NaturalIntelligence/imglab">imglab tool</a> to generate bounding box for our images</li>
  <li>imglab will return a XML file -> Change xml_path in file_annotation.ipynb to extract bounding box infomation</li>
  <li>file_annotation.ipynb will return a CSV files -> Uncomment lines 201-208 and 213-220 to generate positive images and negative images, respectively</li>
</ol>
<br>
## TRAIN MODEL
<ul>
  <li>In order to make the model as simple as possible, I used Pnet in <a href="https://arxiv.org/pdf/1604.02878">MTCNN model</a>. This cause the model quite overfit</li>
</ul>
<img src="https://miro.medium.com/max/1400/1*6xkYymO5qetLLjUt0MYJXg.jpeg" alt="Pnet image">
<br>
## TRACKING ALGORITHM
<ul>
<li>Kanade-Lucas-Tomashi algorithm is used in this project -> Take a look at powerpoint file project_explaination.pptx for more information</li>
</ul>
<br>
## PREFERENCE
<ul>
<li><a href="https://link.medium.com/DT9OTyZWgrb">What Does A Face Detection Neural Network Look Like?</a></li>
<li><a href="https://link.medium.com/mf0efN1Wgrb">How Does A Face Detection Program Work? (Using Neural Networks)</a></li>
</ul>
