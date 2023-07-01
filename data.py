import pandas as pd
import numpy as np
import lightgbm as lgb

global datasets


class Data:
    def __init__(self, base, seed=42):
        self.base = base
        self.train_size = 0.8
        self.valid_size = 0.1
        self.test_size = 0.1
        self.data = None
        self.seed = seed
        self.cols = None
        self.groups = None

    def getLgbData(self, dsets, columns=None):
        if columns is not None:
            return [lgb.Dataset(dset[columns], categorical_feature=None, free_raw_data=False) for dset in dsets]
        else:
            return [lgb.Dataset(dset, categorical_feature=None, free_raw_data=False) for dset in dsets]


    def getTrain(self):
        self.readAndCache()
        return self.train[self.cols]


    def getValid(self):
        self.readAndCache()
        return self.valid[self.cols]

    def getTest(self):
        self.readAndCache()
        return self.test[self.cols]
        
    def readAndCache(self):
        if self.data is None:
            self.data = self.read()
            for i_col in self.getCategorical(): 
                self.data[i_col] = self.data[i_col].astype(str).astype('category')

            np.random.seed(self.seed)
            rnd = np.random.rand(self.data.shape[0])
            self.train = self.data.iloc[rnd < self.train_size]
            self.valid = self.data.iloc[
                (self.train_size <= rnd) & (
                    rnd < (self.train_size+self.valid_size))
            ]
            self.test = self.data.iloc[
                self.train_size+self.valid_size <= rnd
            ]
            if(self.cols is None):
                self.cols = list(self.train.drop(self.targets, axis=1).columns)

    def getCategorical(self):
        if self.data is None:
            raise "data needs to be read first"
        return list(self.data.columns[self.data.dtypes=="object"])

    def getLabels(self, ix):
        self.readAndCache()
        train_y = self.train[self.targets[ix]]
        valid_y = self.valid[self.targets[ix]]
        test_y = self.test[self.targets[ix]]
        return train_y, valid_y, test_y

class CacuData(Data):
    def __init__(self, base, seed=42):
        super().__init__(base, seed)
        self.name = "Communities and Crime Unnormalized"
        self.targets = ['murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
                        'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'ViolentCrimesPerPop', 'nonViolPerPop']
        self.cols = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumKidsBornNeverMar", "PctKidsBornNeverMar", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "OwnOccQrange", "RentLowQ", "RentMedian", "RentHighQ", "RentQrange", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop"]
        self.groups = {
            "race": ["racepctblack",  "racePctWhite",  "racePctAsian",  "racePctHisp"],
            "age": ["agePct12t21",  "agePct12t29",  "agePct16t24",  "agePct65up"],
            "income": ["medIncome",  "pctWWage",  "pctWFarmSelf",  "pctWInvInc",  "pctWSocSec",  "pctWPubAsst",  "pctWRetire",  "medFamInc",  "perCapInc",  "NumUnderPov",  "PctPopUnderPov"],
            "racexincome": ["whitePerCap",  "blackPerCap",  "indianPerCap",  "AsianPerCap",  "OtherPerCap",  "HispPerCap"],
            "education": ["PctLess9thGrade",  "PctNotHSGrad",  "PctBSorMore",  "PctUnemployed",  "PctEmploy",  "PctEmplManu",  "PctEmplProfServ",  "PctOccupManu",  "PctOccupMgmtProf"],
            "family": ["MalePctDivorce",  "MalePctNevMarr",  "FemalePctDiv",  "TotalPctDiv",  "PersPerFam",  "PctFam2Par",  "PctKids2Par",  "PctYoungKids2Par",  "PctTeen2Par",  "PctWorkMomYoungKids",  "PctWorkMom", "NumKidsBornNeverMar", "PctKidsBornNeverMar"],
            "immigration": ["PctForeignBorn", "NumImmig",  "PctImmigRecent",  "PctImmigRec5",  "PctImmigRec8",  "PctImmigRec10",  "PctRecentImmig",  "PctRecImmig5",  "PctRecImmig8",  "PctRecImmig10",  "PctSpeakEnglOnly",  "PctNotSpeakEnglWell"],
            "house": ["householdsize",  "PctLargHouseFam",  "PctLargHouseOccup",  "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",  "PctPersDenseHous",  "PctHousLess3BR",  "MedNumBR",  "HousVacant",  "PctHousOccup",  "PctHousOwnOcc",  "PctVacantBoarded",  "PctVacMore6Mos",  "MedYrHousBuilt",  "PctHousNoPhone",  "PctWOFullPlumb",  "OwnOccLowQuart",  "OwnOccMedVal",  "OwnOccHiQuart", "OwnOccQrange", "RentLowQ",  "RentMedian",  "RentHighQ", "RentQrange", "MedRent",  "MedRentPctHousInc",  "MedOwnCostPctInc",  "MedOwnCostPctIncNoMtg"],
            "homelessness": ["NumInShelters",  "NumStreet"],
            "native": ["PctBornSameState",  "PctSameHouse85",  "PctSameCity85",  "PctSameState85"],
            "police": ["LemasSwornFT",  "LemasSwFTPerPop",  "LemasSwFTFieldOps",  "LemasSwFTFieldPerPop",  "LemasTotalReq",  "LemasTotReqPerPop",  "PolicReqPerOffic",  "PolicPerPop"],
            "racexpolice": ["RacialMatchCommPol",  "PctPolicWhite",  "PctPolicBlack",  "PctPolicHisp",  "PctPolicAsian",  "PctPolicMinor",  "OfficAssgnDrugUnits",  "NumKindsDrugsSeiz",  "PolicAveOTWorked",  "PolicCars",  "PolicOperBudg",  "LemasPctPolicOnPatr",  "LemasGangUnitDeploy",  "LemasPctOfficDrugUn",  "PolicBudgPerPop"],
            "landxpop": ["population",  "numbUrban",  "pctUrban",  "LandArea",  "PopDens",  "PctUsePubTrans"],
        }
    def read(self):
        return pd.read_csv(self.base + '/00211/CommViolPredUnnormalizedData.txt', na_values=['?'], low_memory=False, names=['communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar', 'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ', 'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'ViolentCrimesPerPop', 'nonViolPerPop'])


class FccPmData(Data):
    def __init__(self, base, seed=42):
        super().__init__(base, seed)
        self.name = "PM2.5 Data of Five Chinese Cities"
        self.targets = ['PM']
        self.cols = ['hour', 'season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws', 'precipitation', 'Iprec']

    def read(self):
        cities = ['Beijing', 'Chengdu', 'Guangzhou', 'Shanghai', 'Shenyang']
        city_data = []
        for city in cities:
            cdata = pd.read_csv(self.base + '/00394/'+city +
                                'PM20100101_20151231.csv', low_memory=False)
            pm = [c for c in cdata.columns if c.startswith('PM')]
            cdata['PM'] = cdata[pm].mean(axis=1)
            cdata.drop(pm, inplace=True, axis=1)
            city_data.append(cdata)
        return pd.concat(city_data, axis=0, sort=False)


class SuperconductivityData(Data):
    def __init__(self, base, seed=42):
        super().__init__(base, seed)
        self.name = "Superconductivity"
        self.targets = ['critical_temp']

    def read(self):
        return pd.read_csv(self.base + '/00464/train.csv', low_memory=False)


class GarmentProductivityData(Data):
    def __init__(self, base, seed=42):
        super().__init__(base, seed)
        self.name = "Productivity Prediction of Garment Employees"
        self.targets = ['actual_productivity']

    def read(self):
        return pd.read_csv(self.base + '/00597/garments_worker_productivity.csv', low_memory=False)

datasets = [CacuData, FccPmData, SuperconductivityData, GarmentProductivityData]
