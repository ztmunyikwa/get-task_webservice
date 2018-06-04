#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:15:49 2017

@author: danielrock
"""

#crowdflower processor script.
#pre-steps -- delete unnecessary columns, rename column heads to match rubric.
#cf_dir = "/Users/danielrock/Dropbox (MIT)/econML/Data/rubricCFRuns"
#cf_file = "cf_report_DWAS_input.csv"
#cf_phys_file = "cf_report_DWAS_physical_input.csv"
#task_file = "Tasks to DWAs.xlsx"
#jobTaskRatingFile = "Task Ratings.xlsx"
#jobDataFile = "tasksOccupationsIndustries.csv"

import sys
import pandas as pd
import numpy as np
import scipy.stats as sp
import networkx as nx
import os
from matplotlib import pyplot as plt

def getCenter(questionDF, grouper, center='median'):
    """groups by the task id and then pulls the central tendency"""
    from scipy import stats
    if center=='mode':
        return questionDF.groupby(grouper).agg(lambda x: stats.mode(x)[0][0]).reset_index()
    elif center=='mean':
        return questionDF.groupby(grouper).mean().reset_index()
    else:
        return questionDF.groupby(grouper).median().reset_index()
    
def taskCountAboveThreshold(taskDF,smlMeasure,threshold,grouper, percentile=False):
    """takes a measure of SML and for each grouped item (typically a job), it returns
    the count of tasks in that job with an smlMeasure above the threshold in the function.
    The measure is at the job level. Also returned are total task counts and proportions.
    If percentile is specified, it will convert the threshold to a percentile and run things that way"""
    if percentile:
        threshold = np.percentile(taskDF[smlMeasure], threshold)
    thresTaskCount = taskDF[taskDF[smlMeasure]>threshold][[grouper, smlMeasure]].groupby(grouper).count()
    allTaskCount = taskDF[[grouper, smlMeasure]].groupby(grouper).count()
    outdf = thresTaskCount.join(allTaskCount,lsuffix='a',rsuffix='b').reset_index()
    outdf.rename(columns={smlMeasure+'a':smlMeasure+'_threshold'+str(threshold), smlMeasure+'b':'allTaskCount'}, inplace=True)
    outdf[smlMeasure+'_thresholdProp'+str(threshold)] = outdf[smlMeasure+'_threshold'+str(threshold)]/outdf.allTaskCount
    return outdf

def similarities(df):
    """calculates the similarity score matrix (used for activities)"""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(df)


def wavg(taskDF, grouper, colNames, weightName):
    meanfunc = lambda x: np.average(x[colNames], weights=x[weightName])
    return taskDF.groupby(grouper).apply(meanfunc).reset_index()

def plotSingleSML(OESdetail, smlMeasure, percentile1, sml_label, perc1_label):
    plt.style.use('ggplot')
    plt.scatter(OESdetail[percentile1], OESdetail[smlMeasure])
    plt.title(sml_label+' vs. '+perc1_label)
    plt.xlabel(perc1_label)
    plt.ylabel(sml_label)
    #figsize = plt.rcParams['figure.figsize']
    #print figsize

def plotStackedSML(OESdetail, smlMeasure, percentile1, percentile2, perc1_label, perc2_label, sml_label, color='red'):
    plt.style.use('ggplot')
    plt.figure(1)
    plt.rcParams["figure.figsize"] = [6.0,16.0]
    plt.subplot(211)
    plt.scatter(OESdetail[percentile1],OESdetail[smlMeasure],color=color)
    plt.title(sml_label+" Score vs. "+perc1_label)
    plt.xlabel("Occupational "+perc1_label)
    plt.ylabel(sml_label+" Score")
    plt.subplot(212)
    plt.scatter(OESdetail[percentile2],OESdetail[smlMeasure], color=color)
    plt.title(sml_label+" vs. "+perc2_label)
    plt.xlabel("Occupational "+perc2_label)
    plt.ylabel(sml_label+" Score")
    plt.tight_layout()
    plt.savefig(smlMeasure+".png",dpi=300)
    plt.show()
    
def seabornMultiHist(df, title, xlabel, ylabel):
    """creates multiple histogram plot used in figure 1"""
    plt.style.use('ggplot')
    df.plot(kind='hist',alpha=.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("hist_percentiles.png")
    plt.show()
    
def newAddMeasure(df, existingQs, newQName):
    """define the list of questions which are inputs to a new measure, add them."""
    newMeasure = df[existingQs[0]].copy()
    for q in existingQs[1:]:
        newMeasure += df[q]
    df['q'+newQName] = newMeasure / float(len(existingQs))
    return df
   

if __name__ == '__main__':
    #file read-in
    cf_dir, cf_file, cf_phys_file = sys.argv[1:4]
    os.chdir(cf_dir)
    cf = pd.read_csv(cf_file)
    cf_phys = pd.read_csv(cf_phys_file) #physical questions asked in separate run
    
    #filter to people who understand the task
    cfu = cf[cf.understand>2]
    cfpu = cf_phys[cf_phys.understand>2]
    qfields = ['q'+str(item+1) for item in list(range(21))] + ['dwa_id']
    pqfields = ['q'+str(item+1) for item in [21,22]] + ['dwa_id']
    qs = cfu[qfields] # only the questions and task id
    pqs = cfpu[pqfields]
    qs['variance']=qs[list(qs)[:-1]].var(axis=1) #calculate row-wise variance 
    #not doing the variance filter for the physical questions (only 2 qs)
    qs2 = qs[qs.variance!=0] #drop the zero variance rows. These are people who answered one thing.
    centerQ = getCenter(qs2, 'dwa_id', center='median') #return the median answer for what's left
    centerQp = getCenter(pqs, 'dwa_id', center='median')
    centerQp[['q22','q23']] = 6.0 - centerQp[['q22','q23']] #map physical to bad for ML
    centerQ['qD'] = centerQ[['q15','q16','q17','q18']].max(axis=1) #max of the data questions
    qfields.insert(-1,'qD')
    centerQ = pd.merge(centerQ, centerQp, on='dwa_id')
    qfields = qfields+pqfields[0:2] #now we've merged in the physical-ness
    
    
    #okay now we have the ratings for each activity. Let's join them to the tasks.
    task_file = sys.argv[4]
    tasks = pd.read_excel(task_file) #it's an excel file usually
    tasksDWAs = tasks.merge(centerQ, left_on=['DWA ID'], right_on=['dwa_id'], how='left')
    #average over all of the DWAs in a task to get the average task values.
    #we omit the dwa_id here. It's grouping at the task level.
    taskScores = getCenter(tasksDWAs[[item for item in qfields if item[0]=='q'] + ['Task ID']],'Task ID','mean')
    #merge back and store these to csv
    onlyTasks = tasks[['O*NET-SOC Code','Title','Task ID','Task']].drop_duplicates()
    taskScores2 = taskScores.merge(onlyTasks, on='Task ID')
    taskScores2.to_csv('Tasks_Scores.csv', index=False)
    
    #Now with the task data, we can aggregate to the job level
    jobTaskRatingFile, jobDataFile = sys.argv[5:7]
    jobs = pd.read_excel(jobTaskRatingFile)
    jobsData = pd.read_csv(jobDataFile)
    #get the importance values out for each of the tasks
    jobTask = jobs[jobs['Scale ID']=='IM'][['O*NET-SOC Code','Title','Task ID','Task','Data Value']]
    jobTaskScores = pd.merge(jobTask, taskScores2, on=['Task ID','O*NET-SOC Code','Title','Task'])
    jobTaskScores['Data Value'] = jobTaskScores['Data Value']/5.0 #scale to 0-1
    jobTaskScores['weight'] = jobTaskScores['Data Value'] / jobTaskScores.groupby('O*NET-SOC Code')['Data Value'].transform('sum')
    JTS = jobTaskScores.copy()
    
    newBools = ['DeepLearning']
    JTS = newAddMeasure(JTS, ['q'+str(item) for item in [5,7,10,11,14,19,20]], 'DeepLearning')
    JTSCols=list(JTS)
    for item in newBools:
        JTSCols.insert(-2,'q'+item)
    #weighted average calculations for the job
    for q in JTSCols[5:-1]:
        JTS['w'+q] = JTS[q]*JTS['weight']
    JTSfields = [item for item in list(JTS) if item[0:2]=='wq']
    wJTS = JTS.groupby(['O*NET-SOC Code','Title'])[JTSfields].sum().reset_index()
    #join the calcs to the data we have for the job
    jobScores = pd.merge(wJTS, jobsData[['O*NET-SOC Code','Job Description',\
                                         'Projected Growth (2014-2024)','Projected Job Openings (2014-2024)',\
                                         'Industries']], how='left',on='O*NET-SOC Code').drop_duplicates()
    jobScores.to_csv('Job_Scores.csv', index=False)
    #variance calculations
    JTSV = jobTaskScores.copy() #separate copy for the variances
    vJTS = JTSV.merge(wJTS, on=['O*NET-SOC Code','Title'],how='left')
    for q in [k for k in list(vJTS) if k[0]=='q']: #for all the original question values
        vJTS['dV'+q] = vJTS['weight']*(vJTS[q]-vJTS['w'+q])**2 #dV is for sq. deviation value
    vJTSfields = [item for item in list(vJTS) if item[0:2]=='dV']
    #wvJTS = np.sqrt(vJTS.groupby(['O*NET-SOC Code','Title'])[vJTSfields].sum()).reset_index()
    wvJTS = vJTS.groupby(['O*NET-SOC Code','Title'])[vJTSfields].sum().reset_index()
    jobVarianceScores = pd.merge(wvJTS, jobsData[['O*NET-SOC Code','Job Description',\
                                         'Projected Growth (2014-2024)','Projected Job Openings (2014-2024)',\
                                         'Industries']], how='left',on='O*NET-SOC Code').drop_duplicates()
    jobVarianceScores.to_csv('JobVariance_Scores.csv', index=False)
    
    #merge it all together
    allscores = pd.merge(jobScores,wvJTS,on=['O*NET-SOC Code','Title'])
    allscores.to_csv('JobScores_mean_variance.csv')
    
    #categorizing questions as SML or Measurability
    SML = ['q3','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q19','q20','q21','q22','q23']
    measurability = ['q1','q2','q4','qD']
    #mean and variance for each of SML, measurability.
    #this treats each question equally. In future iterations, weights may be applied.
    allscores['SML'] = (1/float(len(SML)))*allscores[['w'+ item for item in SML]].sum(axis=1)
    allscores['vSML'] = (1/float(len(SML)))*allscores[['dV'+ item for item in SML]].sum(axis=1)
    allscores['sdSML'] = np.sqrt(allscores['vSML'])
    allscores['measure'] = (1/float(len(measurability)))*allscores[['w'+ item for item in measurability]].sum(axis=1)
    allscores['vMeasure'] = (1/float(len(measurability)))*allscores[['dV'+ item for item in measurability]].sum(axis=1)
    allscores['sdMeasure'] = np.sqrt(allscores['vMeasure'])
    allscores['mSML'] = (1/float(len(SML+measurability)))*allscores[['w'+ item for item in SML+measurability]].sum(axis=1)
    allscores['vmSML'] = (1/float(len(SML+measurability)))*allscores[['dV'+ item for item in SML+measurability]].sum(axis=1)
    allscores['sdmSML'] = np.sqrt(allscores['vmSML'])
    allscores['text'] = allscores[['wq15']]
    allscores['image'] = allscores[['wq16']]
    allscores['speech'] = allscores[['wq17']]
    allscores['structured'] = allscores[['wq18']]
    
    
    allscores.to_csv('allscores_SML.csv',index=False)
    mlJobs = pd.merge(allscores, jobsData)
    
    #task-level SML on the basis of averages. Again, weights may be applied in future.
    #these scores are calculated so that we can try other measures of job-level SML
    cQ = centerQ.copy() # for activity-level info
    cQ['mSML']=(1/float(len(SML+measurability)))*cQ[[item for item in SML+measurability]].sum(axis=1)
    taskSML = JTS.copy() #using the unweighted tasks for reorganization
    taskSML['SML'] =  (1/float(len(SML)))*taskSML[[item for item in SML]].sum(axis=1)
    taskSML['measure'] = (1/float(len(measurability)))*taskSML[[item for item in measurability]].sum(axis=1)
    taskSML['mSML'] = (1/float(len(SML+measurability)))*taskSML[[item for item in SML+measurability]].sum(axis=1)
    snipper = lambda x: x[:x.find('.')]
    taskSML['occ code'] = taskSML['O*NET-SOC Code'].apply(snipper)
    #okay now what about jobs with proportions of tasks rated 4 or higher
    jobs4Higher = taskCountAboveThreshold(taskSML,'mSML',4,'Title') #not a ton above 0.3
    #90th-75th-50th percentile version. Reorganization measures
    perc90 = taskCountAboveThreshold(taskSML,'mSML',90,'Title',percentile=True)
    perc90occ = taskCountAboveThreshold(taskSML,'mSML',90,'occ code',percentile=True)
    perc75 = taskCountAboveThreshold(taskSML,'mSML',75,'Title',percentile=True)
    perc50 = taskCountAboveThreshold(taskSML,'mSML',50,'Title',percentile=True)
    perc10 = taskCountAboveThreshold(taskSML,'mSML',10,'Title',percentile=True)
    plotin = pd.DataFrame()
    plotin['90th Percentile'] = perc90['mSML_thresholdProp3.85']
    plotin['75th Percentile'] = perc75['mSML_thresholdProp3.675']
    plotin['50th Percentile'] = perc50['mSML_thresholdProp3.45']
    seabornMultiHist(plotin, '', \
                     "Proportion of Tasks in Occupation with SML Above Percentile",'')
    #BLS Wage Data 2016
    wagesOES = pd.read_excel("OES_2016.xlsx")
    allscores['occ code']=allscores['O*NET-SOC Code'].apply(snipper)
    #just doing the mSML for now
    occScoresSML = allscores[['mSML','vmSML','occ code']].groupby('occ code').mean().reset_index()
    OES = pd.merge(wagesOES, occScoresSML, on=['occ code'])
    OES['sdmSML']=np.sqrt(OES['vmSML']) #variation within tasks
    OESdetail = OES[(OES.group=='detailed')&(OES.naics_title=='Cross-industry')\
                    &(OES.area_title=='U.S.')&(OES.a_median.isin(['*','#'])==False)]
    OESdetail=OESdetail.merge(perc90occ[['occ code', 'mSML_thresholdProp3.85']],on='occ code')
    OESdetail.rename(columns={'mSML_thresholdProp3.85':'mSML_reorg'},inplace=True)
    OESdetail['a_median'] = OESdetail['a_median'].astype(float)
    OESdetail['log_median'] = np.log(OESdetail['a_median'])
    OESdetail['log_mSML_reorg'] = np.log(OESdetail['mSML_reorg']) #proportion above 90%
    #percentiler = lambda ser, x: sp.percentileofscore(ser, x)
    OESdetail['log_median_percentileWage'] = \
        OESdetail['log_median'].apply(lambda x: sp.percentileofscore(OESdetail['log_median'],x))
    OESdetail['tot_emp_percentile'] = OESdetail['tot_emp'].apply(lambda x: sp.percentileofscore(OESdetail['tot_emp'],x))
    OESdetail['wagebill'] = OESdetail['tot_emp']*OESdetail['a_mean']
    OESdetail['wagebill_percentile'] = OESdetail['wagebill'].apply(lambda x: sp.percentileofscore(OESdetail['wagebill'],x))
    OESdetail.to_csv("OESdetail_SML.csv", index=False)
    plotStackedSML(OESdetail,'mSML','log_median_percentileWage','wagebill_percentile','Occupational Log Median Wage Percentile','Occupational Wage Bill Percentile','SML')
    #map of the U.S. - MSA
    mapVals = OES[OES.area_type==2] #only the msas, or states
    mapVals.rename(columns={'area':'FIPS'},inplace=True)
    #we care about area, area_title, job, tot_emp, mSML, vmSML for suitability and re-org potential
    mapValsKeep = mapVals[['FIPS','area_title','tot_emp','mSML','vmSML']]
    mapValsKeep = mapValsKeep[mapValsKeep.tot_emp.isin(['**'])==False]
    mapValsKeep['empWeight'] = mapValsKeep['tot_emp']/mapValsKeep.groupby(['FIPS','area_title'])['tot_emp'].transform('sum')
    mapValsKeep['wmSML'] = mapValsKeep['empWeight']*mapValsKeep['mSML']
    mapValsKeep['wvmSML'] = mapValsKeep['empWeight']*mapValsKeep['vmSML']
    regionStats = mapValsKeep[['FIPS','area_title','wmSML','wvmSML']].groupby(['FIPS','area_title']).sum().reset_index()
    regionStats['wsdmSML'] = np.sqrt(regionStats.wvmSML)
    regionStats.to_csv('regionStats.csv')
    
    OESIndpath="/Users/danielrock/Dropbox (MIT)/Research/datasets/Government/BLS/OESIndcsv"
    OESInd = pd.read_csv(OESIndpath+'/SMLNAICS.csv')
    OESInd = OESInd[OESInd.OCC_GROUP=='detailed']
    OESInd = pd.merge(OESInd, occScoresSML, left_on='OCC_CODE',right_on='occ code')
    OI = OESInd[['NAICS','NAICS_TITLE','TOT_EMP','PCT_TOTAL','mSML']].copy()
    OI['NAICS2'] = OI.apply(lambda x: int(str(x['NAICS'])[0:2]), axis=1)
    OI['TOT_EMP'] = OI.TOT_EMP.astype(float)
    OIT = OI.groupby('NAICS2')['TOT_EMP'].sum().reset_index()
    OIT.rename(columns={'TOT_EMP':'TOTTOTEMP'}, inplace=True)
    OI = OI.merge(OIT, on='NAICS2')
    OI['EMPWEIGHT'] = OI.TOT_EMP/OI.TOTTOTEMP
    OI['EMPWEIGHTSML'] = OI.EMPWEIGHT*OI.mSML
    OII = OI.groupby(['NAICS2']).sum().reset_index()
    OII.to_csv('fastIndustry.csv')
    #The activity network of jobs
    taskActivities = tasks[['Title','DWA ID']].drop_duplicates()
    taskActivities['value']=1
    tA = pd.pivot_table(taskActivities, values='value',index=['Title'],columns=['DWA ID'])
    tA.fillna(0.0,inplace=True)
    netTa = np.dot(tA.as_matrix(),tA.as_matrix().T)
    np.fill_diagonal(netTa,0.0)
    jobNet = pd.DataFrame(netTa,index=tA.index,columns=tA.index).reset_index()
    jobNet.rename(columns={'Title':'job1'},inplace=True)
    meltJobNet = pd.melt(jobNet,id_vars=['job1'],value_vars=list(jobNet)[1:])
    g=nx.from_pandas_dataframe(meltJobNet[meltJobNet.value>0],'job1','Title',['value'])
    tempD = perc90[['Title','mSML_threshold3.85']].set_index('Title')
    #wider = lambda x: (x-tempD['mSML_threshold3.85'].mean())/tempD.mSML.std()
    smlNodeAttr = {key:float(value) for key,value in tempD['mSML_threshold3.85'].to_dict().iteritems()}
    nx.set_node_attributes(g, smlNodeAttr, 'mSML')
    nx.write_gexf(g, "jobNet.gexf")
    
    #the activity network of SML ratings
    acts = centerQ[SML+measurability+['dwa_id']].set_index('dwa_id')
    actsSims = similarities(acts)
    np.fill_diagonal(actsSims,0)
    graphActsSims = nx.from_pandas_adjacency(pd.DataFrame(actsSims, index=acts.index, columns=acts.index))
    acts['mSML'] = (1/float(len(SML+measurability)))*acts[[item for item in SML+measurability]].sum(axis=1)
    smlDict = acts.mSML.to_dict()
    newDict = {key:float(value) for key, value in smlDict.iteritems()}
    nx.set_node_attributes(graphActsSims, newDict, 'SML')
    nx.write_gexf(graphActsSims, "activityNet.gexf")
    
    #fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, sharey=True)
#    sns.distplot(flerp['Occupation SML'],ax=ax1)
#    sns.distplot(flerp['Task SML'],ax=ax2)
#    sns.distplot(flerp['Activity SML'],ax=ax3)
    #top 15 / bottom 15
#    allscores[allscores.mSML>3.72][['Title','mSML']].sort_values(by='mSML',ascending=False)
#    allscores[allscores.mSML<3.22][['Title','mSML']].sort_values(by='mSML',ascending=True)
#    furp['90th Percentile']=perc90['mSML_thresholdProp3.85']

furp['75th Percentile']=perc75['mSML_thresholdProp3.675']
furp['50th Percentile']=perc50['mSML_thresholdProp3.45']
sns.distplot(furp)
furp.plot(kind='hist')
furp.plot(kind='hist',alpha=.5)
furp.plot(kind='hist',alpha=.2)
plt.style.use('ggplot')
furp.plot(kind='hist',alpha=.2)

plt.style.use('ggplot')
furp.plot(kind='hist',alpha=.35)

plt.style.use('ggplot')
furp.plot(kind='hist',alpha=.5)

