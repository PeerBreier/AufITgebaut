from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import cv2
import imutils
import random
from itertools import chain
import copy
from math import sin, cos, radians
import os
from tkinter import *
import Functions
import Classes


def CNN(plan, symbol, pBar, overlap=.4):
    """
    Loads CNN models for 'symbol' and scans 'plan' with CNN models.
    :param plan: 'PlanData' object of plan that should be scanned.
    :param symbol: Symbol ID as string, e.g. 'S001'.
    :param pBar: ProgressBar object.
    :param overlap: Desired overlap of ROIs in planImage.
    :return: no return, results are directly stored in planData.
    """
    # ==== load CNN models ====
    model_number = load_model("CNNs/" + symbol + "/Number.h5")
    model_one = load_model("CNNs/" + symbol + "/One.h5")
    model_two = load_model("CNNs/" + symbol + "/Two.h5")

    # ==== load config file ====
    roi_size = 0

    cfg = open("CNNs/" + symbol + "/config.cnn")
    lines = cfg.readlines()
    for line in lines:
        lin = line.split(' = ')
        if lin[0] == 'roi_size':
            roi_size = int(lin[-1])
            break
    cfg.close()
    step = int((1 - overlap) * roi_size)

    if roi_size == 0:
        print("failed to load config.cnn")
        return

    # ====> Start Scan <====
    startTime = datetime.now()
    img = plan.planImage
    h_img, w_img = img.shape
    num_symb = 0

    # ==== Setup ProgressBar ====
    goal = round((w_img - roi_size) / step)
    pBar.bar.config(maximum=goal)
    pBar.progLabel.config(text='Scanning ' + plan.planID)

    for i in range(1, w_img - roi_size, step):
        Classes.GUI.data.root.update()
        for j in range(1, h_img - roi_size, step):

            roi = img[j:j + roi_size, i:i + roi_size]  # shape (roi_size, roi_size)
            roi = np.expand_dims(roi, 4)  # input image shape must be (*, roi_size, roi_size, 1)
            roi = np.expand_dims(roi, 0)

            sumROI = roi.sum()
            part = (roi_size * roi_size * 0.95) * 255
            if sumROI < part:  # check if ROI is mostly empty
                predicted_number = model_number.predict(roi)
                index = np.argmax(predicted_number)

                # if index == 0: no symbol was detected

                if index == 1:
                    predicted_one = model_one.predict(roi)
                    predicted_one *= roi_size

                    x = int(predicted_one[0][0])
                    y = int(predicted_one[0][1])

                    num_symb += 1
                    plan.results.append([symbol, num_symb, i + x, j + y, 'One'])

                elif index == 2:
                    predicted_two = model_two.predict(roi)
                    predicted_two *= roi_size
                    predicted_two = predicted_two.reshape(len(predicted_two), 2, -1)

                    x1 = int(predicted_two[0][0][0])
                    y1 = int(predicted_two[0][0][1])
                    x2 = int(predicted_two[0][1][0])
                    y2 = int(predicted_two[0][1][1])

                    num_symb += 1
                    plan.results.append([symbol, num_symb, i + x1, j + y1, 'two1'])
                    num_symb += 1
                    plan.results.append([symbol, num_symb, i + x2, j + y2, 'two2'])

        pBar.bar.config(value=i / step)
        pBar.overBar.step(100 * step / (w_img - roi_size))

    plan.scanned.append(symbol)

    timeElapsed = datetime.now() - startTime
    print('\nTime elpased:', round(timeElapsed.total_seconds(), 3), "s")


def rotate(src, angle):
    """
    Rotates src about angle degree in mathematical positive direction.
    :param src: Image array.
    :param angle: Angle in degree.
    :return: Returns rotated image and original heigth and width.
    """
    h, w = src.shape

    if angle in [0, 90, 180, 270, 360]:
        rotated = np.rot90(src, angle/90)
    else:
        size = max(w, h) + 10
        x = round((size - w) / 2)
        y = round((size - h) / 2)

        img = np.zeros((size, size), dtype='uint8')
        img[y:y + h, x:x + w] = src + 1

        rotated = np.around(imutils.rotate(img, angle))
        rotated = cut_image(rotated) - 1

    pos = rotated.shape

    return rotated, h, w, pos[0], pos[1]


def cut_image(img):
    """
    Calculates the minimal boundingbox of symbol on image and cuts off outer pixels.
    :param img: Image array.
    :return: Returns symbol image of minimal size.
    """
    (x, y, w, h) = cv2.boundingRect(img)
    return img[y:y + h, x:x + w]


def distance(x1, y1, x2, y2):
    """
    Calculates euklidean distance of to points.
    :param x1: x-Coordinate of point 1
    :param y1: y-Coordinate of point 1
    :param x2: x-Coordinate of point 2
    :param y2: y-Coordinate of point 2
    :return: Returns distance in pixels (float)
    """
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def find_error(predicted, real):
    """
    Calculates percentage of false predictions
    :param predicted: Array of predicted coordinates
    :param real: Array of real coordinated as generated for CNN input
    :return: Returns percentage of Failure
    """
    num = 0
    err = 0

    for p, r in zip(predicted, real):
        predicted_num = np.argmax(p)
        real_num = np.argmax(r)

        if predicted_num != real_num:
            err += 1

        num += 1

    err /= num

    return err


class TrainSetup:
    """
    Contains all relevant parameters for CNN training.
    """
    def __init__(self):
        self.roi_size = [160]
        self.steps = [30]
        self.resize = [.8, .9, 1.0, 1.1, 1.2]
        self.stretch = [.9, 1.1]
        self.epochs = [50]
        self.filters = [30]
        self.kernel_size = [4]
        self.validation_percent = [.15]
        self.rotation = [-40, 40]

        self.loadSetup()

    def loadSetup(self):
        """
        Loads default parameters from config.cnn file.
        :return:
        """
        cfg = open("config.cnn", "r")
        lines = cfg.readlines()
        self.setPar(lines)
        cfg.close()

    def setPar(self, lines):
        """
        Converts file input into floats/integers and sets TrainSetup parameters.
        :param lines: config file content as list of strings.
        :return: no return, directly sets attributes with 'setattr'.
        """
        for line in lines:
            lin = line.split(' = ')
            par = lin[1].split(',')
            if lin[0] in ['resize', 'stretch', 'validation_percent']:
                par = [*map(float, par)]
            elif lin[0] == '':
                continue
            else:
                par = [*map(int, par)]
            setattr(self, lin[0], par)

    def setStd(self, path="config.cnn"):
        """
        Exports current set of attributes to config.cnn file. -> New default setting.
        :return:
        """
        f = open(path, 'w')
        keys = ['roi_size', 'steps', 'resize', 'stretch', 'epochs', 'filters',
                'kernel_size', 'validation_percent', 'rotation']
        for key in keys:
            val = str(getattr(self, key))[1:-1]
            f.write(key + ' = ' + val + '\n')


class CnnIO:
    """
    Contains input and output for one CNN type or for test data.
    """
    def __init__(self, sym, setup, cnnType, bg, pBar):
        self.cnnType = cnnType
        self.array_sym = []

        self.input = []

        self.output = []

        self.test = NONE

        self._create_sym_array(sym, setup.resize, setup.stretch)
        self._create_input(cnnType, bg, setup.rotation, setup.roi_size[0], pBar)

    def _create_sym_array(self, sym, resize, stretch):
        """
        Creates more variations of the input symbol.
        :param sym: Array image of symbol.
        :param resize: List of resize parameters.
        :param stretch: List of stretch parameters
        :return:
        """
        for size in resize:
            if size == 1.0:
                self.array_sym.append(sym)
            else:
                syms = [cv2.resize(s, (round(size * s.shape[1]),
                                       round(size * s.shape[0]))) for s in sym]
                self.array_sym.append(syms)

        for sy in self.array_sym:
            temp = []
            for s in sy:
                stretch_h = [cv2.resize(s, (round(size * s.shape[1]),
                                            round(s.shape[0]))) for size in stretch]
                stretch_w = [cv2.resize(s, (round(s.shape[1]),
                                            round(size * s.shape[0]))) for size in stretch]

                temp += stretch_h + stretch_w
            sym += temp

        self.array_sym = list(chain.from_iterable(self.array_sym))

    def _create_input(self, cnnType, bg, rot, size, pBar):
        """
        Creates training/testing images for CNN of type 'cnnType'.
        :param cnnType: CNN type as string.
        :param bg: List of background ROIs.
        :param rot: Rotation parameters.
        :param size: ROI size (Integer!).
        :param pBar: ProgressBar object.
        :return:
        """
        self.input = copy.deepcopy(bg)

        startTime = datetime.now()
        goal = len(self.input)
        pBar.bar.config(maximum=goal, value=0)
        pBar.progLabel.config(text='Creating training data for ' + cnnType)

        for i, roi in enumerate(self.input):
            Classes.GUI.data.root.update()
            sym_list = []

            if cnnType == 'num':
                no = random.randint(0, 2)
            elif cnnType == 'one':
                no = 1
            else:
                no = 2

            no_output = np.zeros(3, dtype=float)

            h_sym = no * [0]
            w_sym = no * [0]
            pos_w = no * [0]
            pos_h = no * [0]
            coords = np.array([np.nan, np.nan])

            if no > 1:
                phi = random.randint(rot[0], rot[1])

                for n in range(0, no):
                    extra = random.randint(-5, 5)
                    sym, h_sym[n], w_sym[n], pos_h[n], pos_w[n] = rotate(random.choice(self.array_sym), phi + extra)
                    sym_list.append(sym)

                d_min = [(h_sym[i] + h_sym[i + 1]) / 2 for i in range(no - 1)]
                max_h = max([sym_list[i].shape[0] for i in range(no)])
                max_w = max([sym_list[i].shape[1] for i in range(no)])
                leftover_h = size - max_h - sum(d_min) - 10 * (no - 1)
                leftover_w = size - max_w - sum(d_min) - 10 * (no - 1)

                while leftover_h < 0 or leftover_w < 0:
                    print('deleted', len(sym_list))
                    del sym_list[-1], h_sym[-1], w_sym[-1]
                    no -= 1
                    d_min = [(h_sym[i] + h_sym[i + 1]) / 2 for i in range(no - 1)]
                    leftover_h = size - max_h - sum(d_min) - 10 * (no - 1)
                    leftover_w = size - max_w - sum(d_min) - 10 * (no - 1)

                dh = np.zeros(no, dtype=int)
                dw = np.zeros(no, dtype=int)

                for n in range(0, no - 1):
                    d_h = d_min[n] + random.randint(10, 10 + leftover_h // 2)
                    d_w = d_min[n] + random.randint(10, 10 + leftover_h // 2)

                    theta = random.randint(-35, 35)

                    while 1:
                        temp_h = int(cos(radians(theta + phi)) / cos(radians(theta)) * d_h)
                        dh[(n + 1):] += temp_h
                        temp_w = int(sin(radians(theta + phi)) / cos(radians(theta)) * d_w)
                        dw[(n + 1):] += temp_w

                        if abs(dh[-1]) > size - max_h or abs(dw[-1]) > size - max_w:
                            dh[(n + 1):] -= temp_h
                            dw[(n + 1):] -= temp_w
                            theta = -theta
                        else:
                            break

                seed_h = random.randint(abs(np.min(dh)), size - abs(np.max(dh)) - max_h)
                seed_w = random.randint(abs(np.min(dw)), size - abs(np.max(dw)) - max_w)

                dh += np.array([seed_h])
                dw += np.array([seed_w])
                pos_w = np.array(pos_w)
                pos_h = np.array(pos_h)
                pos_w = pos_w//2
                pos_h = pos_h//2

                coords = np.stack((dw + pos_w, dh + pos_h), axis=-1)
                coords = np.reshape(coords, coords.size)
                for n, h, w in zip(range(no), dh, dw):
                    self._place_mask(roi, sym_list[n], h, w)

            elif no == 1:
                mask, _, _, pos_h[0], pos_w[0] = rotate(random.choice(self.array_sym), random.randint(rot[0], rot[1]))
                seed_h = random.randint(0, size - mask.shape[0])
                seed_w = random.randint(0, size - mask.shape[1])

                pos_w = pos_w[0] // 2
                pos_h = pos_h[0] // 2

                coords = np.stack((seed_w + pos_w, seed_h + pos_h), axis=-1)

                self._place_mask(roi, mask, seed_h, seed_w)

            no_output[no] = 1

            if cnnType == 'num':
                self.output.append(no_output)
            else:
                self.output.append(coords / size)
            if i % 800 == 0:
                pBar.bar.step(800)
                pBar.overBar.step(100 * 800 / goal)
        elapsedTime = datetime.now() - startTime
        print(elapsedTime.total_seconds(), 'sec')

    @staticmethod
    def _place_mask(roi, mask, y_pos, x_pos):
        """
        Places mask of symbol on ROI.
        :param roi: Array image of background ROI.
        :param mask: Array image of symbol
        :param y_pos: y-coordinate of ROI to put upper left corner of mask.
        :param x_pos: x-coordinate of ROI to put upper left corner of mask.
        :return:
        """
        maskZero = np.nonzero(mask == 0)
        maskOne = np.nonzero(mask == 1)

        roi[maskZero[0] + y_pos, maskZero[1] + x_pos] = 255
        roi[maskOne[0] + y_pos, maskOne[1] + x_pos] = 0

    def testInput(self, size):
        """
        Shows some of the created training images along with the respective output.
        :param size: ROI size (Integer!)
        :return: Toplevel window object to destroy it later.
        """
        cnnType = self.cnnType
        t = random.randint(0, len(self.input) - 1)
        img_plot = cv2.cvtColor(self.input[t], cv2.COLOR_GRAY2RGB)
        if cnnType == 'num':
            print(self.output[t])
        elif cnnType == 'one':
            cv2.circle(img_plot, (int(self.output[t][0] * size),
                                  int(self.output[t][1] * size)), 3, (0, 0, 255), 2)
            print(self.output[t])
        else:
            cv2.circle(img_plot, (int(self.output[t][0] * size),
                                  int(self.output[t][1] * size)), 3, (0, 0, 255), 2)
            cv2.circle(img_plot, (int(self.output[t][2] * size),
                                  int(self.output[t][3] * size)), 3, (0, 0, 255), 2)
            print(self.output[t])

        win_cut = Toplevel()
        win_cut.title("Zoom")
        win_cut.resizable(width=False, height=False)
        win_cut.focus_set()
        position = '+0+0'
        win_cut.geometry(position)
        canvas_cut = Canvas(win_cut, width=300, height=300)

        self.test = Functions.resize_image(img_plot, (300, 300))

        canvas_cut.create_image(0, 0, image=self.test, anchor=NW)
        canvas_cut.pack()

        return win_cut

    def test_array(self, resize):
        """
        Prints created symbol array as png.
        :param resize: List of resize parameters.
        :return:
        """
        testS = self.array_sym
        size_i = round(testS[0].shape[1] * (max(resize) + .2)) + 10
        size_j = round(testS[0].shape[0] * (max(resize) + .2)) + 10
        a_i = round(len(testS) / 15 * size_i)
        a_j = round(15 * size_j)
        image = np.full((a_j, a_i), 255, dtype='uint8')
        count = 0
        for i in range(2, a_i - size_i, size_i):
            for j in range(2, a_j - size_j, size_j):
                if count == len(testS):
                    break
                self._place_mask(image, testS[count], j, i)
                count += 1

        cv2.imwrite('./Images/symTestDone.png', image)


class TrainData:
    """
    Contains all relevant information for the training process.
    """
    def __init__(self):
        self.sym = []
        self.sym_test = []

        self.bg = []
        self.bg_test = []

        self.ico = np.zeros((32, 32), dtype='uint8')
        self.test = False

        self.setup = TrainSetup()

        self.num = NONE
        self.one = NONE
        self.two = NONE
        self.numT = NONE
        self.oneT = NONE
        self.twoT = NONE

    def create_bg(self, file, pBar, attr='bg'):
        """
        Creates background ROIs from a big background image.
        :param file: Image location as string.
        :param pBar: ProgressBar object.
        :param attr: Training or testing version.
        :return:
        """
        rsize = self.setup.roi_size[0]
        steps = self.setup.steps[0]

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        row, col = img.shape
        randSize = [int(rsize*rfac) for rfac in [.7, .8, .9, 1.0]]

        goal = round((col - rsize) / steps)
        pBar.bar.config(maximum=goal)
        pBar.progLabel.config(text='Creating ' + attr)

        for j in range(0, col-rsize, steps):
            Classes.GUI.data.root.update()
            for i in range(0, row-rsize, steps):
                size = random.choice(randSize)
                roi = img[i:i + size, j:j + size]
                if roi.shape is not (rsize, rsize, 1):
                    roi = cv2.resize(roi, (rsize, rsize))

                roi = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY)[1]
                roi = np.expand_dims(roi, 3)
                self.__dict__[attr].append(roi)
            pBar.bar.config(value=j / steps)
            pBar.overBar.step(100 * steps / (col - rsize))

    def create_sym_ico(self, new_sym):
        """
        Creates Icon of the new symbol for the ResultListBox.
        :param new_sym: SymbolID as string (e.g. 'S001')
        :return:
        """
        temp = self.sym[0] - 1
        h, w = temp.shape

        if w >= h:
            koeff = w / 32
        else:
            koeff = h / 32

        w = round(w / koeff)
        h = round(h / koeff)

        dw = round(16 - w / 2)
        dh = round(16 - h / 2)

        temp = cv2.resize(temp, (w, h))

        ico = np.full((32, 32), 255, dtype='uint8')
        ico[dh:dh+h, dw:dw+w] = temp

        os.mkdir('./CNNs/' + new_sym)

        cv2.imwrite("./CNNs/" + new_sym + "/Icon.png", ico)

    def _create_model(self, dim, CNN_type):
        """
        Creates CNN model for the training process.
        :param dim: Dimensions of the desired output.
        :param CNN_type: CNN type as string.
        :return: CNN Model.
        """
        filters = self.setup.filters[0]
        kernel_size = self.setup.kernel_size[0]
        roi_size = self.setup.roi_size[0]

        model = Sequential([
            Convolution2D(filters, (kernel_size, kernel_size),
                          input_shape=(roi_size, roi_size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(filters * 2, kernel_size, kernel_size, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(filters * 4, kernel_size, kernel_size, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(filters * 8, kernel_size, kernel_size, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(filters * 16, kernel_size, kernel_size, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dropout(0.2),
            Dense(50, activation='relu'),
        ])

        opt = keras.optimizers.RMSprop(lr=0.0001)

        if CNN_type == 'num':
            model.add(Dense(dim, activation='softmax'))
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        elif CNN_type in ['one', 'two']:
            model.add(Dense(dim))
            model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

        return model

    def _display_train_error(self, err_plot, min_place, symb_name, cnnType):
        """
        Creates error plot and exports it as png.
        :param err_plot: Prediction error of the CNN.
        :param min_place: Index of the smallest err_plot value
        :param symb_name: SymbolID as string.
        :param cnnType: CNN type as string.
        :return:
        """
        x = list(range(1, self.setup.epochs[0] + 1))
        plt.clf()
        plt.plot(x, err_plot, 'b')
        plt.legend()
        plt.title('Training accuracy for' + cnnType + 'symbols')
        plt.xlim(1, self.setup.epochs[0] + 1)

        if cnnType == 'num':
            plt.ylim(0, 10)
        else:
            plt.ylim(0, 20)

        plt.xlabel('Number of epochs')

        if cnnType == 'num':
            plt.ylabel('Error in %')
        else:
            plt.ylabel('Error in px')

        plt.annotate(err_plot[min_place], xy=(min_place, err_plot[min_place] + 1), color='black', fontsize='11')
        plt.savefig("CNNs/" + symb_name + "/" + cnnType + ".png", bbox_inches='tight')

    def generateData(self):
        """
        Starts generation of training data.
        :return:
        """
        pBar = Classes.progressWindow(6, 'Creating training Images')
        self.num = CnnIO(self.sym, self.setup, 'num', self.bg, pBar)
        self.one = CnnIO(self.sym, self.setup, 'one', self.bg, pBar)
        self.two = CnnIO(self.sym, self.setup, 'two', self.bg, pBar)
        self.numT = CnnIO(self.sym_test, self.setup, 'num', self.bg_test, pBar)
        self.oneT = CnnIO(self.sym_test, self.setup, 'one', self.bg_test, pBar)
        self.twoT = CnnIO(self.sym_test, self.setup, 'two', self.bg_test, pBar)
        pBar.destroy()

    def get_sym(self, file, size, attr='sym'):
        """
        Reads image of the specified symbol and stores it as array image in attribute.
        :param file: File location as string.
        :param size: Size of the symbol in a 200dpi picture.
        :param attr: Attribute name as string
        :return:
        """
        for f in file:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img = cv2.threshold(img, 180, 1, cv2.THRESH_BINARY_INV)[1]
            img = cut_image(img)

            h, w = img.shape
            if w >= h:
                koeff = w / size
            else:
                koeff = h / size

            self.__dict__[attr].append(cv2.resize(img, (round(w / koeff), round(h / koeff))))

    def train_CNN_num(self, symb_name):
        """
        Creates the Number of symbol CNN.
        :param symb_name: SymbolID as string.
        :return:
        """
        input_array = np.array(self.num.input)
        input_array_test = np.array(self.numT.input)

        out = np.array(self.num.output)
        model = self._create_model(out.shape[1], 'num')

        coeff = int(self.setup.validation_percent[0] * len(input_array))
        train_input = input_array[coeff:]
        val_input = input_array[:coeff]
        train_output = out[coeff:]
        val_output = out[:coeff]

        min_err = 1000
        min_err_train = 1000
        min_err_test = 1000

        err_epoch = np.zeros(self.setup.epochs[0])

        min_place = -1

        for epoch in range(self.setup.epochs[0]):
            print("\nNumber epoch: ", epoch + 1)

            model.fit(train_input, train_output, nb_epoch=1, validation_data=(val_input, val_output),
                      shuffle=True, verbose=2)
            pred_output = model.predict(val_input)
            pred_output_test = model.predict(input_array_test)

            err = find_error(pred_output, val_output)
            err_test = find_error(pred_output_test, self.numT.output)
            err_both = (err + err_test) / 2

            err_epoch[epoch] = round(err_both * 100, 1)

            print('Acc is: ', 100 - round(err * 100, 2), '%')
            print('Acc Test is: ', 100 - round(err_test * 100, 2), '%')
            print('Acc Both is: ', 100 - round(err_both * 100, 2), '%')

            if err < min_err_train:
                min_err_train = err
            if err_test < min_err_test:
                min_err_test = err_test

            if err_both < min_err:
                min_err = err_both
                min_place = epoch
                model.save("CNNs/" + symb_name + "/Number.h5")

        self._display_train_error(err_epoch, min_place, symb_name, 'num')

    def train_CNN_one(self, symb_name):
        """
        Creates the One symbol CNN.
        :param symb_name: SymbolID as string.
        :return:
        """
        img_array = np.array(self.one.input)
        img_array_test = np.array(self.oneT.input)

        out = np.array(self.one.output)
        out_test = np.array(self.oneT.output)
        model = self._create_model(out.shape[1], 'one')

        coeff = int(self.setup.validation_percent[0] * len(img_array))
        train_input = img_array[coeff:]
        val_input = img_array[:coeff]

        train_output = out[coeff:]
        val_output = out[:coeff]

        min_err = 1000
        min_err_train = 1000
        min_err_test = 1000
        size = self.setup.roi_size[0]
        epo = self.setup.epochs[0]

        err_epoch = np.zeros((len(train_output), epo))
        err_epoch_test = np.zeros((len(self.oneT.output), epo))

        err_plot = np.zeros(epo)
        min_place = -1

        for epoch in range(epo):
            print("\nNumber epoch: ", epoch + 1)
            model.fit(train_input, train_output, nb_epoch=1, validation_data=(val_input, val_output), verbose=2)
            pred_output = model.predict(train_input)
            pred_output_test = model.predict(img_array_test)

            for i in range(0, len(pred_output)):
                err = distance(pred_output[i][0] * size,
                               pred_output[i][1] * size,
                               train_output[i][0] * size,
                               train_output[i][1] * size)
                err_epoch[i, epoch] = err

            for i in range(0, len(pred_output_test)):
                err_test = distance(pred_output_test[i][0] * size,
                                    pred_output_test[i][1] * size,
                                    out_test[i][0] * size,
                                    out_test[i][1] * size)
                err_epoch_test[i, epoch] = err_test

            err_train = float(np.mean(err_epoch[:, epoch]))
            err_test = float(np.mean(err_epoch_test[:, epoch]))
            err_both = float((np.mean(err_epoch[:, epoch]) + np.mean(err_epoch_test[:, epoch])) / 2)

            print('Error is: {}'.format(round(err_train, 1)), 'pixels')
            print('Error Test is: {}'.format(round(err_test, 1)), 'pixels')
            print('Error Both is: {}'.format(round(err_both, 1)), 'pixels')

            err_plot[epoch] = round(err_both, 1)

            if err_train < min_err_train:
                min_err_train = err_train
            if err_test < min_err_test:
                min_err_test = err_test

            if err_both < min_err:
                min_err = err_both
                min_place = epoch
                model.save("CNNs/" + symb_name + "/One.h5")

        self._display_train_error(err_plot, min_place, symb_name, 'one')

    def train_CNN_two(self, symb_name):
        """
        Creates the Two symbol CNN.
        :param symb_name: SymbolID as string.
        :return:
        """
        output = np.array(self.two.output)
        output_test = np.array(self.twoT.output)

        model = self._create_model(output.shape[1], 'two')

        inputt = np.array(self.two.input)
        inputt_test = np.array(self.twoT.input)

        coeff = int(self.setup.validation_percent[0] * len(self.bg))
        train_input = inputt[coeff:]
        test_input = inputt[:coeff]
        train_output = output[coeff:]
        test_output = output[:coeff]

        min_err = 1000
        min_err_train = 1000
        min_err_test = 1000
        size = self.setup.roi_size[0]
        epo = self.setup.epochs[0]

        flipped_train_output = np.array(train_output)
        flipped = np.zeros((len(flipped_train_output), epo))
        flipped_test = np.zeros((len(output_test), epo))
        err_epoch = np.zeros((len(flipped_train_output), epo))
        err_epoch_test = np.zeros((len(output_test), epo))

        err_plot = np.zeros(epo)
        min_place = -1

        for epoch in range(epo):
            print("\nNumber epoch: ", epoch + 1)
            model.fit(train_input, flipped_train_output, nb_epoch=1, validation_data=(test_input, test_output),
                      verbose=2)
            pred_output = model.predict(train_input)
            pred_output_test = model.predict(inputt_test)
            # Flipping the images

            for i, (pred_bboxes, exp_bboxes) in enumerate(zip(pred_output, flipped_train_output)):
                flipped_exp_bboxes = np.concatenate([exp_bboxes[2:], exp_bboxes[:2]])

                mse = np.mean(np.square(pred_bboxes - exp_bboxes))
                mse_flipped = np.mean(np.square(pred_bboxes - flipped_exp_bboxes))

                dist = distance(pred_bboxes[0] * size,
                                pred_bboxes[1] * size,
                                exp_bboxes[0] * size,
                                exp_bboxes[1] * size) + distance(pred_bboxes[2] * size,
                                                                 pred_bboxes[3] * size,
                                                                 exp_bboxes[2] * size,
                                                                 exp_bboxes[3] * size)

                dist_flipped = distance(pred_bboxes[0] * size,
                                        pred_bboxes[1] * size,
                                        flipped_exp_bboxes[0] * size,
                                        flipped_exp_bboxes[1] * size) + distance(pred_bboxes[2] * size,
                                                                                 pred_bboxes[3] * size,
                                                                                 flipped_exp_bboxes[2] * size,
                                                                                 flipped_exp_bboxes[3] * size)

                if mse_flipped < mse:  # you can also use iou or dist here
                    flipped_train_output[i] = flipped_exp_bboxes
                    flipped[i, epoch] = 1
                    err_epoch[i, epoch] = dist_flipped / 2.
                else:
                    err_epoch[i, epoch] = dist / 2.

            for i, (pred_bboxes, exp_bboxes) in enumerate(zip(pred_output_test, output_test)):
                flipped_exp_bboxes = np.concatenate([exp_bboxes[2:], exp_bboxes[:2]])
                mse = np.mean(np.square(pred_bboxes - exp_bboxes))
                mse_flipped = np.mean(np.square(pred_bboxes - flipped_exp_bboxes))

                dist = distance(pred_bboxes[0] * size,
                                pred_bboxes[1] * size,
                                exp_bboxes[0] * size,
                                exp_bboxes[1] * size) + distance(pred_bboxes[2] * size,
                                                                 pred_bboxes[3] * size,
                                                                 exp_bboxes[2] * size,
                                                                 exp_bboxes[3] * size)

                dist_flipped = distance(pred_bboxes[0] * size,
                                        pred_bboxes[1] * size,
                                        flipped_exp_bboxes[0] * size,
                                        flipped_exp_bboxes[1] * size) + distance(pred_bboxes[2] * size,
                                                                                 pred_bboxes[3] * size,
                                                                                 flipped_exp_bboxes[2] * size,
                                                                                 flipped_exp_bboxes[3] * size)

                if mse_flipped < mse:  # you can also use iou or dist here
                    output_test[i] = flipped_exp_bboxes
                    flipped_test[i, epoch] = 1
                    err_epoch_test[i, epoch] = dist_flipped / 2.
                else:
                    err_epoch_test[i, epoch] = dist / 2.

            err_train = float(np.mean(err_epoch[:, epoch]))
            err_test = float(np.mean(err_epoch_test[:, epoch]))
            err_both = (err_train + err_test) / 2

            print('Flipped {} % of all elements'.format(round(float(np.mean(flipped[:, epoch])) * 100., 2)))
            print('Error Train is : {}'.format(round(err_train, 1)), 'pixels')
            print('Error Test is : {}'.format(round(err_test, 1)), 'pixels')
            print('Error Both is : {}'.format(round(err_both, 1)), 'pixels')

            err_plot[epoch] = np.around(err_both, 1)

            if err_train < min_err_train:
                min_err_train = err_train
            if err_test < min_err_test:
                min_err_test = err_test

            if err_both < min_err:
                min_err = err_both
                min_place = epoch
                model.save("CNNs/" + symb_name + "/Two.h5")

        self._display_train_error(err_plot, min_place, symb_name, 'two')
