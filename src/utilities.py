from numpy import concatenate, empty, newaxis, zeros, squeeze
from nifty8 import LinearOperator, DomainTuple, UnstructuredDomain, makeField
from resolve import PolarizationSpace, ms2observations, IRGSpace
from datetime import datetime


class Expander_00DD(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1], domain[2]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((x0, x0, xD, xD), axis=0)
        else:
            x = x.val
            xD = x[2,:,:,:] + x[3,:,:,:]
            res = xD

        return makeField(self._tgt(mode), res)


class Expander_0D0D(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1], domain[2]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((x0, xD, x0, xD), axis=0)
        else:
            x = x.val
            xD = x[1,:,:,:] + x[3,:,:,:]
            res = xD

        return makeField(self._tgt(mode), res)


class Expander_D0D0(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1], domain[2]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((xD, x0, xD, x0), axis=0)
        else:
            x = x.val
            xD = x[0,:,:,:] + x[2,:,:,:]
            res = xD

        return makeField(self._tgt(mode), res)


class Expander_DD00(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1], domain[2]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((xD, xD, x0, x0), axis=0)
        else:
            x = x.val
            xD = x[0,:,:,:] + x[1,:,:,:]
            res = xD

        return makeField(self._tgt(mode), res)



class Expander_R000(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([PolarizationSpace(('RR','RL','LR','LL')), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xR = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((xR, x0, x0, x0), axis=0)
        else:
            x = x.val
            xR = x[0,:,:]
            res = xR

        return makeField(self._tgt(mode), res)



class Expander_0R00(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([PolarizationSpace(('RR','RL','LR','LL')), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xR = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((x0, xR, x0, x0), axis=0)
        else:
            x = x.val
            xR = x[1,:,:]
            res = xR

        return makeField(self._tgt(mode), res)



class Expander_00R0(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([PolarizationSpace(('RR','RL','LR','LL')), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xR = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((x0, x0, xR, x0), axis=0)
        else:
            x = x.val
            xR = x[2,:,:]
            res = xR

        return makeField(self._tgt(mode), res)



class Expander_000R(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([PolarizationSpace(('RR','RL','LR','LL')), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xR = x[newaxis, :]
            x0_ = zeros(x.shape)
            x0 = x0_[newaxis, :]
            res = concatenate((x0, x0, x0, xR), axis=0)
        else:
            x = x.val
            xR = x[3,:,:]
            res = xR

        return makeField(self._tgt(mode), res)


class Expander_NNPP(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            res = concatenate((-1*xD, -1*xD, xD, xD), axis=0)
        else:
            x = x.val
            xD = -1*x[0,:,:] -1*x[1,:,:] + x[2,:,:] + x[3,:,:]
            res = xD

        return makeField(self._tgt(mode), res)


class Expander_PNPN(LinearOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([UnstructuredDomain(4), domain[0], domain[1]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):

        if mode == self.TIMES:
            x = x.val
            xD = x[newaxis, :]
            res = concatenate((xD, -1*xD, xD, -1*xD), axis=0)
        else:
            x = x.val
            xD = x[0,:,:] -1*x[1,:,:] + x[2,:,:] -1*x[3,:,:]
            res = xD

        return makeField(self._tgt(mode), res)


def get_BeginTime_UTC(config, data_path=None):
    cfg_observation = config["observation"]
    if data_path == None:
        data_path = cfg_observation["data_path"]
    spectral_window = cfg_observation.getint("spectral_window")
    polarizations = cfg_observation["polarizations"]

    obs = ms2observations(data_path, "DATA", True, spectral_window, polarizations)[0]
    # Difference between MDJ(1858/11/17/00:00:00) and UTC(1970/01/01/00:00:00) is 3506716800 seconds
    BeginTime_UTC_list = []
    for ii in range(len(obs.time)):
        BeginTime_UTC_list.append(datetime.utcfromtimestamp(obs.time[ii] - 3506716800))
    timestamp = obs.time - 3506716800
    return BeginTime_UTC_list, timestamp


def get_files_in_folder(folder):
    import os

    folder = os.path.expanduser(folder)
    dirpath, _, files = next(os.walk(folder))
    return [os.path.join(dirpath, ff) for ff in files]



class get_singlefreqsky(LinearOperator):
    def __init__(self, domain, freq_number):
        self.freq_number = freq_number
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make([domain[0], domain[1], IRGSpace([domain[2].coordinates[freq_number]]), domain[3]])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        

        if mode == self.TIMES:
            x = x.val
            res = x[:,:,self.freq_number:self.freq_number+1,:,:]

        else:
            x = x.val
            x0 = zeros(x.shape)

            res_ = []
            for ii in range(int(self._domain[2].size)-1):
                res_.append(x0)
            res_.insert(self.freq_number, x)
            res = concatenate(res_, axis=2)

        
        return makeField(self._tgt(mode), res)