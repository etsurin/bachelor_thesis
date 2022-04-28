from pynput.keyboard import Key,Listener

def on_press(key):
    if key == Key.left:
        print('you press left')

    else:
        print('you press other buttons')

    # return False

def on_release(key):

    if key == Key.left:

        print('you release left')
    
    else:
        print('you release other buttons')

    return False    

#监听键盘按键

with Listener(on_press=on_press,on_release=on_release) as listener:

    listener.join()

#停止监视

    listener.stop()
