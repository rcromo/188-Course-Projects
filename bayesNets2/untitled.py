    def getHouseValCorrespToPos(x, y):
        width, height = gameState.data.layout.width, gameState.data.layout.height

        if x < width / 2.0 and y < height / 2.0:
            return BOTTOM_LEFT_VAL
        elif x > width / 2.0 and y < height / 2.0:
            return BOTTOM_RIGHT_VAL
        elif x < width / 2.0 and y > height / 2.0:
            return TOP_LEFT_VAL
        elif x > width / 2.0 and y > height / 2.0:
            return TOP_RIGHT_VAL
        else:
            print "Error: x and y inputs into getHouseValCorrespToPos are invalid."
            return None

    for housePos in gameState.getPossibleHouses():
        for obsPos in gameState.getHouseWalls(housePos):
            obsVar = OBS_VAR_TEMPLATE % obsPos
            obsFactor = bn.Factor([obsVar], [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], bayesNet.variableDomainsDict())
            for assignment in obsFactor.getAllPossibleAssignmentDicts():
                probRed = 0
                probBlue = 0
                probNone = 0
                if assignment[FOOD_HOUSE_VAR] == getHouseValCorrespToPos(obsPos[0], obsPos[1]):
                    probRed = PROB_FOOD_RED
                    probBlue = 1 - PROB_FOOD_RED
                    probNone = 0
                elif assignment[GHOST_HOUSE_VAR] == getHouseValCorrespToPos(obsPos[0], obsPos[1]):
                    probRed = PROB_GHOST_RED
                    probBlue = 1 - PROB_GHOST_RED
                    probNone = 0
                else: 
                    probRed = 0
                    probBlue = 0
                    probNone = 1

                if assignment[obsVar] == BLUE_OBS_VAL:
                    obsFactor.setProbability(assignment, probBlue)
                elif assignment[obsVar] == RED_OBS_VAL:
                    obsFactor.setProbability(assignment, probRed)
                else: # when assignment[obsVar] == NO_OBS_VAL:
                    obsFactor.setProbability(assignment, probNone)

            bayesNet.setCPT(obsVar, obsFactor)
