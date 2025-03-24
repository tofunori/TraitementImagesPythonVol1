quarto render --cache --to html --output-dir ./docs

quarto convert 00-PriseEnMainPython.qmd
marimo convert 00-PriseEnMainPython.ipynb  -o ./marimo/00-PriseEnMainPython.py
#jupyter nbconvert 00-PriseEnMainPython.ipynb  --to pdf --output-dir pdfs
#jupyter nbconvert 00-PriseEnMainPython.ipynb --to latex --output-dir latex
mv 00-PriseEnMainPython.ipynb ./notebooks/

quarto convert index.qmd
#jupyter nbconvert index.ipynb  --to pdf --output-dir pdfs
#jupyter nbconvert index.ipynb --to latex --output-dir latex

quarto convert 00-auteurs.qmd
#jupyter nbconvert 00-auteurs.ipynb  --to pdf --output-dir pdfs
#jupyter nbconvert 00-auteurs.ipynb --to latex --output-dir latex

quarto convert references.qmd
#jupyter nbconvert references.ipynb  --to pdf --output-dir pdfs
#jupyter nbconvert references.ipynb --to latex --output-dir latex

quarto convert 01-ImportationManipulationImages.qmd
marimo convert 01-ImportationManipulationImages.ipynb  -o ./marimo/01-ImportationManipulationImages.py
#jupyter nbconvert 01-ImportationManipulationImages.ipynb  --execute --to pdf --output-dir pdfs
#jupyter nbconvert 01-ImportationManipulationImages.ipynb --execute --to latex --output-dir latex
mv 01-ImportationManipulationImages.ipynb ./notebooks/

quarto convert 02-RehaussementVisualisationImages.qmd
marimo convert 02-RehaussementVisualisationImages.ipynb  -o ./marimo/02-RehaussementVisualisationImages.py
#jupyter nbconvert 02-RehaussementVisualisationImages.ipynb  --execute --to pdf --output-dir pdfs
#jupyter nbconvert 02-RehaussementVisualisationImages.ipynb --execute --to latex --output-dir latex
mv 02-RehaussementVisualisationImages.ipynb ./notebooks/

quarto convert 03-TransformationSpectrales.qmd
marimo convert 03-TransformationSpectrales.ipynb  -o ./marimo/03-TransformationSpectrales.py
#jupyter nbconvert 03-TransformationSpectrales.ipynb  --execute --to pdf --output-dir pdfs
#jupyter nbconvert 03-TransformationSpectrales.ipynb --execute --to latex --output-dir latex
mv 03-TransformationSpectrales.ipynb ./notebooks/

quarto convert 04-TransformationSpatiales.qmd
marimo convert 04-TransformationSpatiales.ipynb  -o ./marimo/04-TransformationSpatiales.py
#jupyter nbconvert 04-TransformationSpatiales.ipynb  --execute --to pdf --output-dir pdfs
#jupyter nbconvert 04-TransformationSpatiales.ipynb --execute --to latex --output-dir latex
mv 04-TransformationSpatiales.ipynb ./notebooks/

quarto convert 05-ClassificationsSupervisees.qmd
marimo convert 05-ClassificationsSupervisees.ipynb  -o ./marimo/05-ClassificationsSupervisees.py
#jupyter nbconvert 05-ClassificationsSupervisees.ipynb  --execute --to pdf --output-dir pdfs
#jupyter nbconvert 05-ClassificationsSupervisees.ipynb --execute --to latex --output-dir latex
mv 05-ClassificationsSupervisees.ipynb ./notebooks/


# #quarto render 04-TransformationSpatiales.qmd --to docx  --no-execute --output-dir ./docx
quarto render --cache --no-clean --to docx --output-dir ./docx
mv -f ./docx/Traitement-d-images-satellites-avec-Python.docx .

quarto render --profile production --cache --no-clean --to pdf --output-dir ./pdf
mv -f ./pdf/Traitement-d-images-satellites-avec-Python.pdf ./docs


git add .
git commit -m 'new content'
git push