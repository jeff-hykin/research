1. Get the jt command
`pip install jupyterthemes`

1. Set the theme by the running command
`jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T` <br>
source: [https://github.com/dunovank/jupyter-themes](https://github.com/dunovank/jupyter-themes)<br>
the command can be customized<br>
jt  [-h] [-l]<br>
   [-t THEME]<br>
   [-f MONOFONT]<br>
   [-fs MONOSIZE]<br>
   [-nf NBFONT]<br>
   [-nfs NBFONTSIZE]<br>
   [-tf TCFONT]<br>
   [-tfs TCFONTSIZE]<br>
   [-dfs DFFONTSIZE]<br>
   [-m MARGINS]<br>
   [-cursw CURSORWIDTH]<br>
   [-cursc CURSORCOLOR]<br>
   [-vim]<br>
   [-cellw CELLWIDTH]<br>
   [-lineh LINEHEIGHT]<br>
   [-altp]<br>
   [-altmd]<br>
   [-altout]<br>
   [-P]<br>
   [-T]<br>
   [-N]<br>
   [-r]<br>
   [-dfonts]<br>

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
