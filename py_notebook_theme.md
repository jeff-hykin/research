1. Get the jt command
`pip install jupyterthemes`

1. Set the theme by the running command
`jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T`
source: [https://github.com/dunovank/jupyter-themes](https://github.com/dunovank/jupyter-themes)
the command can be customized
jt  [-h] [-l]
   [-t THEME]
   [-f MONOFONT]
   [-fs MONOSIZE]
   [-nf NBFONT]
   [-nfs NBFONTSIZE]
   [-tf TCFONT]
   [-tfs TCFONTSIZE]
   [-dfs DFFONTSIZE]
   [-m MARGINS]
   [-cursw CURSORWIDTH]
   [-cursc CURSORCOLOR]
   [-vim]
   [-cellw CELLWIDTH]
   [-lineh LINEHEIGHT]
   [-altp]
   [-altmd]
   [-altout]
   [-P]
   [-T]
   [-N]
   [-r]
   [-dfonts] 

3. Restart your notebook server
`python3 -m notebook`
Refresh the webpage

1. The normal browser print messes everything up <br>
   
   So: put this as the first cell in your notebook<br>
   (prevents scrolling output)
   ```javascript
   %%javascript
   IPython.OutputArea.prototype._should_scroll = ()=>false
   ```
   Then get a scrolling screen capture extension like this one: [GoFullPage](https://chrome.google.com/webstore/detail/gofullpage-full-page-scre/fdpohaocaechififmbbbbbknoalclacl)
   
   Use that to take a scrolling screenshot.
   For that extension in particular
   - go to the settings 
   - find "paper size" and
   - choose "full image" for max asethetic
