from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from kivy.uix.button import Button
from functools import partial


class PongGame(Widget):
    pass


class PongApp(App):
 
    def disable(self, instance, *args):
 
        instance.disabled = True
 
    def update(self, instance, *args):
 
        instance.text = "I am Disabled!"
 
    def build(self):
 
        mybtn = Button(text="Click me to disable")
 
        mybtn.bind(on_press=partial(self.disable, mybtn))
 
        mybtn.bind(on_press=partial(self.update, mybtn))
        pb = ProgressBar(max=1000) 
        pb.value = 750
        return pb


if __name__ == '__main__':
    PongApp().run()

