# -*- coding: utf-8 -*-
"""

@author: charl
"""


def ShotsToArrays(shots):

    import pandas as pd
    import numpy as np
    from pandas.io.json import json_normalize    

    #strip out free kicks/pens
    shots = shots[shots.play_pattern != 'Other']
    
    expanded_shots = json_normalize(shots['shot']) # expanded shot is produced for each shot in order, so can reset index and append shots.id onto expanded_shots

    shots = shots.reset_index(drop=True)
    expanded_shots = expanded_shots.reset_index(drop=True)
    expanded_shots = pd.concat([shots, expanded_shots], axis = 1)
    expanded_shots[['loc_x','loc_y']] = pd.DataFrame(expanded_shots.location.tolist(), index= expanded_shots.index)
    expanded_shots[['endloc_x', 'endloc_y', 'endloc_z']] =  pd.DataFrame(expanded_shots.end_location.tolist(), index= expanded_shots.index)
    
    
    freeze_frames = expanded_shots.loc[:,['freeze_frame','id']]
    
    list_of_ffFrames = []
    for frame in freeze_frames.itertuples(index=False):
        if frame[0] != frame[0]: # using the nan != nan trick to find find nans in column of dicts
            continue
        
        df = pd.DataFrame(frame[0])
        df['id'] = frame[1]
        df_player = json_normalize(df['player']) # expand the dict cols, append, then drop the originals
        df_position = json_normalize(df['position'])
        df = df.reset_index(drop=True)
        df_player = df_player.reset_index(drop=True)
        df_position = df_position.reset_index(drop=True)
        df = pd.concat([df, df_player,df_position], axis = 1)
        df = df.drop(['player','position'], axis = 1)
        list_of_ffFrames.append(df)
        
    # can concat as all frames in list have same columns  
    ff_frame = pd.concat(list_of_ffFrames)
    ff_frame[['loc_x','loc_y']] = pd.DataFrame(ff_frame.location.tolist(), index= ff_frame.index)
    ff_frame = ff_frame.drop('location', axis = 1)  
    ff_frame.columns.values[[1, 2, 3, 4, 5]] = ['shot_id', 'player_id', 'player_name', 'pos_id', 'pos_name'] 
    
    shots = expanded_shots.drop(['location', 'end_location', 'index', 'minute', 'second', 'type', 'related_events', 'shot','freeze_frame', 'key_pass_id'], axis = 1)
    
    shots = shots[shots['body_part.name'] != 'Head']
    
    # strip out shots taken from >40 x dist
    shots = shots[shots.loc_x >= 81] # 81 to 120 gives 40 units
    
    shots_ref= shots[['id',
              'player',
              'play_pattern',
              'body_part.name',
              'outcome.name',
              'statsbomb_xg',
              'team',
              'loc_x',
              'loc_y']]
    
    shots = shots_ref[['id',
              'loc_x',
              'loc_y',
              'outcome.name',
              'statsbomb_xg',
              'team']]
    

    # change outcome to 0,1 and rename column
    shots.loc[shots['outcome.name'] == 'Goal', 'outcome.name'] = 1
    shots.loc[shots['outcome.name'] != 1, 'outcome.name'] = 0
    shots = shots.rename(columns={'outcome.name': 'shot_outcome'})
    
    # subtract 81 from all x locs so val becomes 0 to 40
    shots.loc[:,'loc_x'] = shots.loc[:,'loc_x'] - 81
    
    # round x, y to integer
    shots.loc[:,'loc_x'] = shots.loc[:,'loc_x'].astype(int)
    shots.loc[:,'loc_y'] = shots.loc[:,'loc_y'].astype(int)
    
    # add channel column and set values to 0 as shot xy will be channel 0
    shots.loc[:,'Channel']= 0
    
    # trim FF data
    ff_df=ff_frame[['teammate',
                 'pos_name',
                  'loc_x',
                  'loc_y',
                  'shot_id']]
    
    
    # get shot and FF frames to match so can combine by renaming teammate column to channel and id to shotID
    ff_df = ff_df.rename(columns={'teammate': 'Channel', 'shot_id': 'id'})
    
    # subtract 81 from all x locs so val becomes 0 to 40
    ff_df.loc[:,'loc_x'] = ff_df.loc[:,'loc_x'] - 81
    
    # round x, y to integer
    ff_df.loc[:,'loc_x'] = ff_df.loc[:,'loc_x'].round()
    ff_df.loc[:,'loc_y'] = ff_df.loc[:,'loc_y'].round()
    
     
    # reomve temamate (channel)  = true so only opp players left
    ff_df = ff_df[ff_df.Channel != True]
    
    # change old false to channel = 2
    ff_df.loc[ff_df.Channel == False, 'Channel'] = 2
    
    # change channel to 1 where position_name = GK
    ff_df.loc[ff_df.pos_name == 'Goalkeeper', 'Channel'] = 1
    
    # drop position_name
    ff_df=ff_df[['Channel',
               'loc_x',
                 'loc_y',
                 'id']]
    
    # concat FF and shot arrays
    combined = pd.concat([shots,ff_df], sort = True)
    
    
    # keep only the frames that are in the shot df, i.e. get rid of header FFs
    
    combined = combined[combined['id'].isin(shots['id'])]
    
    combined = combined.set_index('id') # make index the id

    channel0 = combined[combined['Channel'] == 0] 
    channel0.loc[:,['loc_x','loc_y', 'shot_outcome']]=channel0.loc[:,['loc_x','loc_y', 'shot_outcome']].astype(int)
    
    channel1 = combined[combined['Channel'] == 1].iloc[:,0:3]
    channel1.loc[:,['loc_x','loc_y']]=channel1.loc[:,['loc_x','loc_y']].astype(int)
    
    channel2 = combined[combined['Channel'] == 2].iloc[:,0:3]
    channel2.loc[:,['loc_x','loc_y']]=channel2.loc[:,['loc_x','loc_y']].astype(int)
    
    # delete all rows (defenders) that are outside frame - shot is in frame so defs behind
    channel2 = channel2[channel2['loc_x']>=0]
    
    # delete all rows (GKs) that are outside frame - shot is in frame so defs behind
    channel1 = channel1[channel1['loc_x']>=0]
    
    channel0 = channel0.sort_index() # very important to sort here
    channel1 = channel1.sort_index()
    channel2 = channel2.sort_index()
    
    # identify the shot ids where there is no opp defs and no GK in frame
    NoOpp = channel0[~channel0.index.isin(channel2.index)]
    NoGk = channel0[~channel0.index.isin(channel1.index)]
    
    outcomes = channel0.loc[:,['shot_outcome']].sort_index() # must sort by index as concat output will be sorted - which is required anyway
    xG = channel0.loc[:,['statsbomb_xg']].sort_index()
    team = channel0.loc[:,['team']].sort_index()
    loc_x = channel0.loc[:,['loc_x']].sort_index()
    loc_y = channel0.loc[:,['loc_y']].sort_index()
    
    aux_shot_data = pd.concat([outcomes, xG, team, loc_x, loc_y],axis = 1)
    
    channel0_locs = channel0.groupby(level=0).agg({'loc_x':list, 'loc_y':list})
    channel1_locs = channel1.groupby(level=0).agg({'loc_x':list, 'loc_y':list})
    channel2_locs = channel2.groupby(level=0).agg({'loc_x':list, 'loc_y':list})
    
    # function to take locations and convert to a 2D binary image
    def make_matrix(r):
        m = np.zeros((40,80,1)) # create array of zeros and populate the 1s
        try:
            for x,y in zip(r[0],r[1]):
                m[x,y-1,0] = 1
            for x,y in zip(r[2],r[3]):
                m[x,y-1,1] = 1
            for x,y in zip(r[4],r[5]):
                m[x,y-1,2] = 1
        except: # ID is missing for this channel
            pass
        return m
    
    # create lists of key val pairs that match shot id to a binarized freeze frame for each channels
    
    channel0_dict = dict(enumerate(channel0_locs.apply(make_matrix, axis = 1).to_list()))
    channel0_list = [{k: v[1]} for k,v in zip(channel0.index,channel0_dict.items())]# replace keys with the actual shot id

    channel1_dict = dict(enumerate(channel1_locs.apply(make_matrix, axis = 1).to_list()))
    channel1_list = [{k: v[1]} for k,v in zip(channel1.index,channel1_dict.items())]
    
    channel2_dict = dict(enumerate(channel2_locs.apply(make_matrix, axis = 1).to_list()))
    channel2_list = [{k: v[1]} for k,v in zip(channel2.index.unique(),channel2_dict.items())] # .unique() very important here since channel 2 has >1 rows per index
    
    # add the arrays of zeros for the shots that have no keeper and no defenders
    
    NoOpp_list = []
    for sid in list(NoOpp.index):
        item = {sid: np.zeros((40,80,1))}
        NoOpp_list.append(item)
    
    # concat no opp list with channel 2 
    channel2_list = channel2_list+NoOpp_list
    # re-sort list
    channel2_list = sorted(channel2_list, key=lambda d: list(d.keys()))

    # same for GK and channel 1
    NoGk_list = []
    for sid in list(NoGk.index):
        item = {sid: np.zeros((40,80,1))}
        NoGk_list.append(item)
    
    
    # concat no GK list with channel 1
    channel1_list = channel1_list+NoGk_list
    # re-sort list
    channel1_list = sorted(channel1_list, key=lambda d: list(d.keys()))
    
    # now all channels are populated and ordered by shot id we can remove the ids and stack to make a 4D array    

    c0=[]
    c1=[]
    c2=[]
    
    for d in channel0_list:
        val = list(d.values())[0]
        c0.append(val)
        
    for d in channel1_list:
        val = list(d.values())[0]
        c1.append(val)

    for d in channel2_list:
        val = list(d.values())[0]
        c2.append(val)     
    
    
    output = np.squeeze(np.stack([c0,c1,c2], axis = 3)) # stack on new axis to create 4D array, squeeze to remove final dimension of 1
    return output, aux_shot_data