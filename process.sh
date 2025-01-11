quarto render --no-cache
quarto convert 01-ImportationManipulationImages.qmd
mv 01-ImportationManipulationImages.ipynb ./notebooks/

quarto convert 02-RehaussementVisualisationImages.qmd
mv 02-RehaussementVisualisationImages.ipynb ./notebooks/

quarto convert 03-TransformationSpectrales.qmd
mv 03-TransformationSpectrales.ipynb ./notebooks/

quarto convert 04-TransformationSpatiales.qmd
mv 04-TransformationSpatiales.ipynb ./notebooks/
quarto render 04-TransformationSpatiales.qmd --to docx  --no-execute --output-dir ./docx

git add .
git commit -m 'new content'
git push