# -*- coding: utf-8 -*-
"""
@author: charl
"""

def LoadEvents(event_type): # one of ['ball_receipts','blocks','carrys','clearances',
                            #'dispossesseds','foul_committeds','interceptions','miscontrols',
                            #'passes','pressures','shots']

    from statsbombpy import public, sb
    import pandas as pd
    from itertools import chain
    from pandas.io.json import json_normalize
    
    
    #### CREATE COMPETITIONS TABLE ####
    
    comps = public.competitions()
    comp_frame = pd.DataFrame.from_dict(comps, orient = 'index') 
    comp_frame['comp_seas'] = comp_frame['competition_id'].astype(str) +'_'+comp_frame['season_id'].astype(str) # concat comp and seas ids and set as index
    comp_frame = comp_frame.set_index('comp_seas') 
    
    
    ###################################
    
    #### CREATE MATCHES TABLE ####
    
    matches_dictlist=[] # returns list of dict of dicts
    for i in range(0, len(comp_frame.index)):
        matches = public.matches(comp_frame['competition_id'][i], comp_frame['season_id'][i])
        matches_dictlist.append(matches)
    
    
    match_list = [] # convert to list of list of dicts
    for dicts in matches_dictlist:
        list_of_dicts = [value for value in dicts.values()]
        match_list.append(list_of_dicts)
        
    
    #chain.from_iterable is chaining together each list within a super list. So list of list of dicts -> list of dicts
    # create a dataframe out of a single list of all match dictionaries
    match_frame = pd.DataFrame(list(chain.from_iterable(match_list))) 
    
    cols = ['competition',
            'season',
            'home_team',
            'away_team',]
    
    list_of_exp_frames = []
    for col in cols:
        df = json_normalize(match_frame[col])
        list_of_exp_frames.append(df)
        
    expanded_match = pd.concat(list_of_exp_frames, axis = 1)    
    match_frame = pd.concat([match_frame, expanded_match], axis = 1)
    match_frame['comp_seas'] =  match_frame['competition_id'].astype(str) +'_'+match_frame['season_id'].astype(str) # concat comp and seas ids for use as foreign key
    
    match_frame = match_frame.loc[:,~match_frame.columns.duplicated()] # drop duplicated cols (both)
    match_frame = match_frame.drop(['competition',
                                    'season',
                                    'home_team',
                                    'away_team',
                                    'match_status',
                                    'metadata',
                                    'competition_stage',
                                    'stadium',
                                    'referee',
                                    'home_team_group',
                                    'away_team_group',
                                    'managers'], axis = 1)   
    
    
    #women's only
    #match_frame = match_frame[match_frame['home_team_gender'] == 'female']    
        
    #men's only
    #match_frame = match_frame[(match_frame['home_team_gender'] == 'male') ]     
              
    match_ids = list(match_frame["match_id"])
    
    ##### CREATE EVENTS TABLE ####
    
    # this process only needs to be done when re-loading to the events table, the resultant data is approx 1.3GB as of June 2020
    
    grouped_event_list = []
    for match_id in match_ids:
        grouped_events = sb.events(match_id=match_id, split=True)   # takes about 25 mins to run with 850+ games, even using local json files as oppsed to API calls
        grouped_event_list.append(grouped_events)
    
    
    #select the event group required and a list of dataframes will be returned
    #shots_list = [i["shots"] for i in grouped_event_list]
    
    def event_type_agg(event_type):
        event_list = [i[event_type] for i in grouped_event_list]
        event_list = [x for x in event_list if isinstance(x, list) == False] # remove empty list entries
        events = pd.concat(event_list, sort = False)
        return events
    
    event_types = ['ball_receipts',
                    'blocks',
                    'carrys',
                    'clearances',
                    'dispossesseds',
                    'foul_committeds',
                    'interceptions',
                    'miscontrols',
                    'passes',
                    'pressures',
                    'shots']
    events = {}
    for event_type in event_types:
        events[event_type] = event_type_agg(event_type)
        events[event_type]
    
    ##################################
    
    output = events[event_type]
    
    return output