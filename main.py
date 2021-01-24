#Follow instructions here: https://hackernoon.com/build-a-custom-trained-object-detection-model-with-5-lines-of-code-y08n33vi

#numpy version 1.19.3
import numpy as nm
from detecto import core, utils, visualize
import torch
import matplotlib.pyplot as plt
import pyautogui as pag
import time
import math
#import discord
from twilio.rest import Client

#cuda was installed
#pytorch was installed ;;;; pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


def testStuff():
    x = 0
# Example
# image = utils.read_image('fruit.jpeg')
# model = core.Model()
# labels, boxes, scores = model.predict_top(image)
# visualize.show_labeled_image(image, boxes, labels)

# print(torch.has_cuda)

# Train the dataSet
# dataset = core.Dataset()
# model = core.Model(['cow', 'goblin', 'calf', 'dairy cow'])
# model.fit(dataset)

# Test
# image = utils.read_image('Screenshot (8).png')
# predictions = model.predict(image)

# predictions format: (labels,boxes,scores)
# labels, boxes, scores = predictions

# print(labels)
# print(boxes)
# print(scores)

# visualize the predictions
# visualize.show_labeled_image(image, boxes, labels)

# save the model!!!!
# model.save('runeScapeAI.pth')

# load saved model
# model = core.Model.load('runeScapeAI.pth', ['cow', 'goblin', 'calf', 'dairy cow'])


# Capture Screen
# image = pag.screenshot();
# image2 = utils.read_image('Screenshot (8).png')

# Run the model and time the prediction
# startTime = time.time()
# predictions = model.predict(image2)
# predictionSpeed = time.time() - startTime

# Save the parameters
# labels, boxes, scores = predictions
# visualize.show_labeled_image(image2, boxes, labels)

# zipping the prediction values together
# params = zip(labels, boxes)
# params = list(params)
# print(params)

# sorting the list alphabetically
# res = sorted(params, key=lambda x: x[0])
# show time
# print(res)

# I was planning to build discord integration so you could control it remotely but I ended up using a remote session with google 
# and adjusting my player with the runescape mobile app!!!
client = Client('TWILIO CLIENT', 'TWILIO CLIENT')


def sendHelp():
    client.messages.create(to="YOUR PHONE NUMBER",
                           from_="TWILIO PHONE NUMBER",
                           body="Help")

########################################################################################################################
#                                           Classes for Final Project                                                  #
########################################################################################################################

#set pause for pause
pag.PAUSE = 0

#To label new dataSets open Terminal and enter 'labelImg' to start the labeling program
#AI DataSet
combatDS = 'DataSet/'
inventoryDS = 'InventoryDS'

#AI Classes
combatClasses = ['cow', 'goblin', 'calf', 'dairy cow', 'incombat', 'incombat block', 'incombat hit', 'sand crab']
cowsClasses = ['cow', 'goblin', 'calf', 'dairy cow']
#AI filePaths
combatFilePath = 'runeScapeAI.pth'
combat2FilePath = 'combatAI2'
inventoryFilePath = 'inventoryAI.pth'


#Middle of Screen
xMid = (int)(1919/2)
yMid = (int)(1079/2) -30 #Adjusted to be at center of avatar

# (FULLSCREEN) - Click on the middle far right of the mini map
mmapFarRightX = 1907
mmapFarRightY = 135
# (FULLSCREEN) - Click on the middle far left of the mini map
mmapFarLeftX = 1728
mmapFarLeftY = 135


#Propmt the user to start program
pag.alert("Start")

def trainModel(modelName, dataSet, classifiers):
    # Train the dataSet
    dataset = core.Dataset((dataSet + '/'))
    model = core.Model(classes=classifiers)
    model.fit(dataset)
    model.save(modelName)

def loadModel(filePath, classes):
    #load saved model
    model = core.Model.load(filePath, classes)
    return model

def testModel(model, imagePath):
    image = utils.read_image(imagePath)
    predictions = model.predict(image)
    labels, boxes, scores = predictions
    visualize.show_labeled_image(image, boxes, labels)



#TENSOR VALUES ACCURATELY REPRESENT PIXEL LOCATIONS ON SCREEN LOL
#X: 0 - 1919 (Left -> Right)
#Y: 0 - 1079 (Top -> Bottom)
#Box Tensors are returned as xMin, yMin, xMax, yMax

def getScreen():
    return pag.screenshot()

def clickOnTarget(coordinates):
    #Extract Values
    xMin = coordinates[0]
    yMin = coordinates[1]
    xMax = coordinates[2]
    yMax = coordinates[3]
    #Get Coordinates
    xMiddle = xMin + ((xMax - xMin) / 2)
    yMiddle = yMin + ((yMax - yMin) / 2)

    #Move mouse and click
    pag.moveTo(xMiddle, yMiddle, duration=0.2)
    time.sleep(0.3)
    pag.leftClick()

    #center mouse
    pag.moveTo(xMid, yMid, duration=0.2)

def removePredictions(targets):
    index = []
    list = targets[2].numpy()
    for i in range(len(list)):
        if(list[i] < .85):
            index.append(i)
    return index


def getPredictions(model):
    image = getScreen()
    predictions = model.predict(image)
    index = removePredictions(predictions)
    labels, boxes, scores = predictions
    #res = zipValuesAndSort(labels, boxes)
    params = zip(labels, boxes)
    params = list(params)
    for i in range(len(index)):
        params.pop(index[len(index)-1-i])
    return params

def getClosestCow(targets):
    index = 0
    closest = targets[0]
    for i in range(len(targets)):
        if (targets[i][0] == 'calf' or targets[i][0] == 'cow'):
            targetxMin = targets[i][1].numpy()[0]
            targetyMin = targets[i][1].numpy()[1]
            closestxMin = closest[1].numpy()[0]
            closestyMin = closest[1].numpy()[1]

            targetDiag = math.pow((targetxMin - xMid), 2) + math.pow((targetyMin - yMid), 2)
            closestDiag = math.pow((closestxMin - xMid), 2) + math.pow((closestyMin - yMid), 2)
            if(targetDiag < closestDiag):
                closest = targets[i]
    return closest

def getClosestCrab(targets):
    index = 0
    if(len(targets) == 0):
        return None
    else:
        closest = targets[0]
        for i in range(len(targets)):
            if (targets[i][0] == 'sand crab'):
                targetxMin = targets[i][1].numpy()[0]
                targetyMin = targets[i][1].numpy()[1]
                closestxMin = closest[1].numpy()[0]
                closestyMin = closest[1].numpy()[1]

                targetDiag = math.pow((targetxMin - xMid), 2) + math.pow((targetyMin - yMid), 2)
                closestDiag = math.pow((closestxMin - xMid), 2) + math.pow((closestyMin - yMid), 2)
                if(targetDiag < closestDiag):
                    closest = targets[i]
        return closest

def getCombat(model):
    targets = getPredictions(model)
    combat = False
    for i in range(len(targets)):
        if(targets[i][0] == 'incombat hit' or targets[i][0] == 'incombat block'):
            combat = True
            break
    return combat


def mainCows():
    model = loadModel(combatFilePath, cowsClasses)
    targets = getPredictions(model)
    #Bot timer
    botLife = time.time()
    #Official Loop
    flag = False
    while(time.time() - botLife < 9000000000):
        targets = getPredictions(model)
        #Avoid timing out of server
        if(len(targets) == 0):
            pag.rightClick()
            time.sleep(.25)
            pag.leftClick()
            time.sleep(1)
        else:
            closest = getClosestCow(targets)
            clickOnTarget(closest[1].numpy())
            time.sleep(2.5)

def killCrabs():
    model = loadModel(combat2FilePath, combatClasses)
    targets = getPredictions(model)
    noCrabFoundForAwhile = False
    foundCrabNoAggro = False
    timeOut = time.time()
    nullEncounterTick = 0
    # Official Loop
    flag = False
    # Timeout after 5 mins
    while (noCrabFoundForAwhile != True and ((time.time() - timeOut) < 450)):
        targets = getPredictions(model)
        closest = getClosestCrab(targets)
        # If no crab found
        if(closest == None ):
            noCrabFoundForAwhile = True
        # If crabs are present but they are not aggro'd to the player -> Move then return
        elif(nullEncounterTick > 10):
            noCrabFoundForAwhile = False
            #Move around
            for i in range(3):
                pag.moveTo(mmapFarRightX, mmapFarRightY, duration=0.3)
                pag.leftClick()
                time.sleep(11)
            for i in range(3):
                pag.moveTo(mmapFarLeftX, mmapFarLeftY, duration=0.3)
                pag.leftClick()
                time.sleep(11)
            nullEncounterTick = 0
        else:
            noCrabFoundForAwhile = False
            clickOnTarget(closest[1].numpy())
            inCombat = time.time()
            timeOut = time.time()
            while(time.time() - inCombat < 4):
                inCombatFlag = getCombat(model)
                if(inCombatFlag):
                    inCombat = time.time()
                    nullEncounterTick = 0
                else:
                    nullEncounterTick += 1
                    time.sleep(1)

    # no crabs found and time out achieved -> Send Discord alert
    sendHelp()

def getCoor():
    for i in range(10):
        print(pag.position())
        time.sleep(1)

def toBankFromMine():
    # Starting outside of castle right mine
    pag.moveTo(1847, 49, duration=0.5)
    pag.leftClick()
    time.sleep(15)
    print('1')

    pag.moveTo(1819, 46, duration=0.5)
    pag.leftClick()
    time.sleep(15)
    print('2')

    pag.moveTo(1792, 46, duration=0.5)
    pag.leftClick()
    time.sleep(15)
    print('3')

    pag.moveTo(1735, 94, duration=0.5)
    pag.leftClick()
    time.sleep(15)
    print('4')

    pag.moveTo(1731, 137, duration=0.5)
    pag.leftClick()
    time.sleep(15)
    print('5')

    pag.moveTo(1812, 169, duration=0.5)
    pag.leftClick()
    time.sleep(7)
    print('6')

    # click on banker
    pag.moveTo(951, 654, duration=0.5)
    time.sleep(1)
    pag.rightClick()
    time.sleep(1)
    print('7')

    # Check this
    pag.moveTo(923, 701, duration=0.5)
    pag.leftClick()
    time.sleep(1)
    print('8')

    # dump inventory
    pag.moveTo(1054, 785, duration=0.5)
    pag.leftClick()
    time.sleep(1)
    print('9')

    # Return to Mine
    # Walk out bank
    pag.moveTo(1899, 91, duration=0.5)
    pag.leftClick()
    time.sleep(15)

    pag.moveTo(1860, 212, duration=0.5)
    pag.leftClick()
    time.sleep(15)

    pag.moveTo(1872, 204, duration=0.5)
    pag.leftClick()
    time.sleep(15)

    pag.moveTo(1831, 219, duration=0.5)
    pag.leftClick()
    time.sleep(15)

    pag.moveTo(1788, 206, duration=0.5)
    pag.leftClick()
    time.sleep(10)

def mine():
    for i in range(14):
        pag.moveTo(914, 541, duration=0.2)
        time.sleep(5)
        pag.leftClick()
        pag.moveTo(952, 498, duration=0.2)
        time.sleep(5)
        pag.leftClick()

def mineNBank():
    start = time.time()
    while(time.time() - start < 90000000):
        mine()
        toBankFromMine()

def smithDarts():
    #Face north, zoom all the way out, and look birds eye
    #starting at first anvil
    pag.moveTo(903, 271, duration=0.5)
    time.sleep(0.5)
    pag.leftClick()

    time.sleep(4.5)

    # select ore (first item in bank)
    pag.moveTo(612, 148, duration=0.5)
    time.sleep(0.5)
    pag.leftClick()

    # select Anvil
    pag.moveTo(1080, 892, duration=0.5)
    time.sleep(0.5)
    pag.leftClick()

    time.sleep(4.5)

    #select dart tips
    pag.moveTo(952, 296, duration=0.5)
    time.sleep(0.5)
    pag.leftClick()

    #Sleep and repeat
    time.sleep(80)

    pass

def dartGrind():
    #1500 bars to start
    start = time.time()
    for i in range(900):
        smithDarts()
    sendHelp()
    print((time.time() - start))

def spamSpell():
    for i in range(14700):
        pag.leftClick()
        time.sleep(2.5)


def test():
    # Test Stuff
    # sendHelp()
    getCoor()
    # toBankFromMine()
    # mineNBank()
    pass

# spamSpell()
dartGrind()
# test()
# mainCows()
# killCrabs()




