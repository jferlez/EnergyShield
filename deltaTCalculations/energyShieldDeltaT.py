import os
import sys
import numpy as np
import pickle
import math

thismodule = sys.modules[__name__]

with open('energyShieldPyLUT.p','rb') as fp:
    lut = pickle.load(fp)
assert len(lut) > 0, 'ERROR: LUT is empty'
lut.sort(key=(lambda x: x['offs']))

offsets = np.array([ lutObj['offs'] for lutObj in lut], dtype=np.float64)

paramList = ['rbar', 'sigma', 'vmax', 'lr', 'deltaFMax', 'betaMax']
for p in paramList:
    setattr(thismodule, p, lut[0][p])
    assert all([lutObj[p] == getattr(thismodule, p) for lutObj in lut]), f'ERROR: LUT contains different values for problem parameter \'{p}\'...'

def rmin(xi):
    return rbar/(sigma*np.cos(xi/2) + 1 - sigma)

def deltaT(r=0, xi=0, beta=0):
    assert xi >= -np.pi and xi <= np.pi, f'ERROR: xi = {xi} is outside of the range [-pi, pi]'
    assert beta >= -betaMax and beta <= betaMax, f'ERROR: beta = {beta} is outside of the range [-{betaMax}, {betaMax}]'

    validOffsets = np.nonzero( (rmin(xi) + offsets - r) > 0 )[0]
    if len(validOffsets) > 0:
        offsetIdx = np.min(validOffsets)
        offset = lut[offsetIdx]['offs']
    else:
        return 0
    
    xiIncrement = lut[offsetIdx]['xiIncrement']
    betaIncrement = lut[offsetIdx]['betaIncrement']

    xiIndex = int(np.floor((xi+np.pi)/xiIncrement))
    assert xiIndex >=0 and xiIndex < len(lut[offsetIdx]['xiPoints']), 'INTERNAL ERROR: xiIndex out of range'
    betaIndex = int(np.floor((beta+betaMax)/betaIncrement))
    assert betaIndex >=0 and betaIndex < len(lut[offsetIdx]['betaPoints']), 'INTERNAL ERROR: betaIndex out of range'

    return lut[offsetIdx]['lut'][xiIndex, betaIndex]