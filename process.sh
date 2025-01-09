quarto render --no-cache
quarto convert 01-ImportationManipulationImages.qmd
mv 01-ImportationManipulationImages.ipynb ./notebooks/
quarto render 01-ImportationManipulationImages.qmd --to docx  --no-execute --output-dir ./docx
quarto convert 02-RehaussementVisualisationImages.qmd
mv 02-RehaussementVisualisationImages.ipynb ./notebooks/
quarto render 02-RehaussementVisualisationImages.qmd --to docx  --no-execute --output-dir ./docx
quarto convert 03-TransformationSpectrales.qmd
mv 03-TransformationSpectrales.ipynb ./notebooks/
quarto render 03-TransformationSpectrales.qmd --to docx  --no-execute --output-dir ./docx

quarto convert 04-TransformationSpatiales.qmd
mv 04-TransformationSpatiales.ipynb ./notebooks/
quarto render 04-TransformationSpatiales.qmd --to docx  --no-execute --output-dir ./docx

git add .
git commit -m 'new content'
git push