import time
import threading
import multiprocessing
import multiprocessing.process


if __name__ == "__main__":
    be, fe = multiprocessing.Pipe()

    def recv():
        while True:
            res = fe.recv()
            print(res)
            if res is None:
                break
            msg, data = res
            if msg == "text":
                print(data)
            if msg == "audio":
                print(data)
                time.sleep(1)

    def send_t1():
        for i in range(10):
            time.sleep(0.1)
            be.send(("text", "t"))
        be.send(None)

    def send_t2():
        for i in range(10):
            time.sleep(0.3)
            be.send(("audio", "a"))
        be.send(None)

    t1 = threading.Thread(target=send_t1, args=())
    t2 = threading.Thread(target=send_t2, args=())

    t1.start()
    t2.start()

    recv()

    t1.join()
    t2.join()

    be.close()
    fe.close()
