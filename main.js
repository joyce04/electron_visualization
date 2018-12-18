const { app, session, BrowserWindow } = require('electron');
const path = require('path');

const EXEC_MODULE = 'run_flask'
const PY_MODULE = 'run_app'
const PY_FOLDER = 'py_source'

const guessPackaged = () => {
    if (process.platform === 'win32') {
        return require('fs').existsSync(fullPath = path.join(__dirname, EXEC_MODULE + '.exe'))
    } else {
        return require('fs').existsSync(path.join(__dirname, EXEC_MODULE))
    }
}

const getScriptPath = () => {
    if (!guessPackaged()) {
        return path.join(__dirname, PY_FOLDER, PY_MODULE + '.py')
    }
    if (process.platform === 'win32') {
        return path.join(__dirname, EXEC_MODULE + '.exe')
    }
    return path.join(__dirname, EXEC_MODULE)
}

// Keep a global reference of the mainWindowdow object, if you don't, the mainWindowdow will
// be closed automatically when the JavaScript object is garbage collected.
var mainWindow = null;
var subpy = null;

function createWindow() {
    // Create the browser mainWindow
    mainWindow = new BrowserWindow({
        width: 1300,
        height: 950,
        // transparent: true, // transparent header bar
        // icon: __dirname + '/web/resource/icon.png',
        // fullscreen: true,
        // opacity:0.8,
        // darkTheme: true,
        // frame: false,
        resizeable: true,
        show: false
    });

    const ses = mainWindow.webContents.session.clearCache(function () { });

    // Load the index page
    mainWindow.loadFile('./web/init.html')

    // Open the DevTools.
    mainWindow.webContents.openDevTools();

    // Emitted when the mainWindow is closed.
    mainWindow.on('closed', () => {
        // Dereference the mainWindow object
        mainWindow = null;
    });

    mainWindow.once('ready-to-show', () => {
        mainWindow.show()
    })
}

function addAppEventListeners() {
    // disable menu
    app.on('browser-window-created', (e, window) => {
        window.setMenu(null);
    });

    // ------- app terminated
    app.on('window-all-closed', () => {
        // On macOs it is common for application and their icon
        // to stay active until the user quits explicitly with Cmd + q
        if (process.platform !== 'darwin') {
            app.quit()
        }
    });

    app.on('quit', () => {
        // kill the python server on exit
        subpy.kill('SIGINT');
        subpy = null;
    });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser mainWindow.
// Some APIs can only be used after this event occurs.
app.on('ready', () => {
    let script = getScriptPath()
    console.log(script)

    if (guessPackaged()) {
        subpy = require('child_process').execFile(script)
    } else {
        subpy = require('child_process').spawn('python3', [script])
    }

    if (subpy != null) {
        console.log('child process success on port 5000')
        console.log('child process pid is ' + subpy.pid)
    }

    createWindow();
    addAppEventListeners();
});