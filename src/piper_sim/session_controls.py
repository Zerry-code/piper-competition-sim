"""Viewer hotkeys only queue commands; simulation mutations stay on the main thread."""
class RestartRound(Exception):
    pass


class SessionControls:
    def __init__(self,mode='fruits'):
        self.mode=mode
        self.request=None
        self.paused=False

    def key(self,keycode):
        # GLFW function keys do not enter the Windows IME composition stream.
        if keycode==294:self.request=('replay',self.mode)  # F5
        elif keycode==295:self.request=('random',self.mode) # F6
        elif keycode==296:self.paused=not self.paused      # F7
        key=chr(keycode).upper() if 0<=keycode<256 else ''
        if key=='R':self.request=('replay',self.mode)
        elif key=='N':self.request=('random',self.mode)
        elif key=='1':self.request=('switch','blocks')
        elif key=='2':self.request=('switch','fruits')
        elif key=='P':self.paused=not self.paused

    def check(self):
        if self.request is not None:raise RestartRound()
