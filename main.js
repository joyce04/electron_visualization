const {app, BrowserWindow} = require('electron')

function createWindow(){
  window = new BrowserWindow({width: 800, height: 600})
  window.loadFile('index.html');
}
app.on('ready', createWindow)

app.on('window-all-closed', ()=>{
  // On macOs it is common for application and their icon
  // to stay active until the user quits explicitly with Cmd + q
  if(process.platform !== 'darwin'){
    app.quit()
  }
})
