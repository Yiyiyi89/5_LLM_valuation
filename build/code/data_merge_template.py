
import os
import re
import pandas as pd
import numpy as np
import recordlinkage

import warnings
from toolkit import clean_column,NgramsOverlap,determine_entity
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
# from toolkit import ngrams_overlap

from sklearn.feature_extraction.text import CountVectorizer


warnings.filterwarnings("ignore")
from config import
code_path = os.getcwd()
parent_path = os.path.dirname(code_path)
temp_data_path = os.path.join(parent_path, "data", "temp")
raw_data_path = os.path.join(parent_path, "data", "raw")
output_data_path = os.path.join(parent_path, "data", "output")


# only run it once , just in case
# revelio = pd.read_csv(os.path.join(raw_data_path, "mba_students_education.csv.gz"),usecols=['university_name'],compression='gzip').drop_duplicates().dropna().reset_index(drop=True)
# revelio = revelio.sort_values('university_name').reset_index(drop=True)
# revelio['university_id'] = revelio.index+1
# revelio.to_csv(os.path.join(temp_data_path, "revelio_university_lookup.csv"),index=False)
# del(revelio)
# revelio_uni_lookup = pd.read_csv(os.path.join(temp_data_path, "revelio_university_lookup.csv"))

if __name__ == "__main__":    
    """
    merge revelio and ipeds
    using: university_name
    """
    revelio_uni_lookup = pd.read_csv(os.path.join(temp_data_path, "revelio_university_lookup.csv"))
    revelio = pd.read_csv(os.path.join(raw_data_path, "mba_students_education.csv.gz"),usecols=['university_name','degree'],compression='gzip').drop_duplicates().dropna().reset_index(drop=True)
    revelio = revelio[revelio['degree']=='MBA']
    revelio = revelio.drop_duplicates().dropna().reset_index(drop=True).drop(columns=['degree'],axis=1)
    revelio = revelio.merge(revelio_uni_lookup, how='left', on='university_name')
    
    ipeds = pd.read_stata(os.path.join(raw_data_path, "University_Directory.dta"))
    ipeds = ipeds[['unitid', 'instnm']].drop_duplicates().dropna().reset_index(drop=True)
    
    
    df_queries = revelio[['university_id','university_name']]
    df_candidates = ipeds[['unitid', 'instnm']]

    query_id = 'university_id'
    candidate_id = 'unitid'
    query_column = 'university_name'
    candidate_column = 'instnm'
    
    entity_list = ['university','school','college','center','centre','academy','institute']
    directions_list = [
        'northern', 'southern', 'eastern', 'western',
        'northeast', 'southeast', 'northwest', 'southwest',
    ]
    
    """
    clean data
    """
    # unify data type
    df_queries[[query_id]] = df_queries[[query_id]].astype(int)
    df_candidates[[candidate_id]] = df_candidates[[candidate_id]].astype(int)
    df_queries[[query_column]] = df_queries[[query_column]].astype(str)
    df_candidates[[candidate_column]] = df_candidates[[candidate_column]].astype(str)

    
    # add entity, contains uni, sch, col, ctr, entity is uni, sch, col, ctr, if not, it is 'other', if one or more, it is 'multi'

    df_queries['entity'] = determine_entity(df_queries,query_column, entity_list)
    df_candidates['entity'] = determine_entity(df_candidates,candidate_column, entity_list)
    df_queries['direction'] = determine_entity(df_queries,query_column, directions_list)
    df_candidates['direction'] = determine_entity(df_candidates,candidate_column, directions_list)
    
    keywords_list = ['of', 'the', 'and', 'at','lnc','main campus'] + entity_list
    keywords_dict = {
        'technology': 'tech', 'technical': 'tech', 'state': 'stat',
        'community': 'comm', 'christian': 'christ', 'north': 'abcde',
        'south': 'fghij', 'east': 'klmno', 'west': 'pqrst', 'saint': 'st',
        'aampm': 'a m', 'aampt': 'a t'
    }
    df_queries[f'{query_column}_clean'] = clean_column(df_queries[query_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    df_candidates[f'{candidate_column}_clean'] = clean_column(df_candidates[candidate_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    
    """
    only use name similarity
    """
    indexer = recordlinkage.Index()
    indexer.block(left_on=['entity', 'direction'], right_on=['entity', 'direction'])
    candidate_links = indexer.index(df_queries, df_candidates)
    
    indexer = NgramsOverlap(query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean', n=4)
    ngram_overlap = indexer.index(df_queries, df_candidates)
    
    candidate_links = candidate_links.intersection(ngram_overlap)
    compare = recordlinkage.Compare(n_jobs=max(os.cpu_count()*0.9, 1))
    compare.string(f"{query_column}_clean", f"{candidate_column}_clean", method='levenshtein', label='name_similarity')
    features = compare.compute(candidate_links, df_queries, df_candidates).reset_index()
    features = features.rename(columns={'level_0': 'query_index', 'level_1': 'candidate_index'})
    features = features.sort_values('name_similarity', ascending=False).drop_duplicates(subset='query_index').drop_duplicates(subset='candidate_index')
    df = pd.merge(df_queries, features, how='inner', left_on=df_queries.index,right_on='query_index').merge(df_candidates, how='inner', left_on='candidate_index',right_on=df_candidates.index)
    df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id).drop_duplicates(subset=candidate_id)
    
    ## multithread version
    # chunk_size = max(1000, len(df_queries) // os.cpu_count())
    # num_chunks = (len(df_queries) + chunk_size - 1) // chunk_size
    # df_queries_chunks = np.array_split(df_queries, num_chunks)

    # def process_chunk(df_queries_chunk):
    #     global df_candidates,query_id,candidate_id,query_column,candidate_column
    #     # block 1: with 5 n-grams overlap
    #     indexer = NgramsOverlap(query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean', n=4)
    #     ngram_overlap = indexer.index(df_queries, df_candidates)
    #     # ngram_overlap = ngrams_overlap(df_queries=df_queries_chunk,df_candidates=df_candidates, 
    #     #                             query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean',
    #     #                             n=5)
    #     # block 2: with the same entity and direction
    #     indexer = recordlinkage.Index()
    #     indexer.block(left_on=['entity', 'direction'], right_on=['entity', 'direction'])
    #     candidate_links = indexer.index(df_queries_chunk, df_candidates)
    #     # get intersection
    #     candidate_links = candidate_links.intersection(ngram_overlap)
    #     compare = recordlinkage.Compare(n_jobs=max(os.cpu_count()*0.9, 1))
    #     compare.string(f"{query_column}_clean", f"{candidate_column}_clean", method='levenshtein', label='name_similarity')
    #     features = compare.compute(candidate_links, df_queries, df_candidates).reset_index()
    #     features = features.rename(columns={'level_0': 'query_index', 'level_1': 'candidate_index'})
    #     features = features.sort_values('name_similarity', ascending=False).drop_duplicates(subset='query_index')
    #     df = pd.merge(df_queries_chunk, features, how='inner', left_on=df_queries_chunk.index,right_on='query_index').merge(df_candidates, how='inner', left_on='candidate_index',right_on=df_candidates.index)
    #     df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id)
    #     return df


    # with tqdm_joblib(tqdm(desc="Processing chunks", total=len(df_queries_chunks))):
    #     df_list = Parallel(n_jobs=-1)(
    #         delayed(process_chunk)(chunk) for chunk in df_queries_chunks
    #     )
    
    # df = pd.concat(df_list)
    # df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id)
    
    df_round1 = df[df['name_similarity']>0.9][[query_id,candidate_id,query_column,candidate_column,'name_similarity']]
    df_round1.to_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_round1_match.csv.gz'), 
                index=False,
                compression='gzip')
    print("round1 done!",flush=True)



















    """
    bring in bschool into ipeds
    """
    
    bschool = pd.read_csv(os.path.join(raw_data_path, "US_Business_schools.csv"))
    bschool['id'] = bschool.index + 1
    ipeds = pd.read_stata(os.path.join(raw_data_path, "University_Directory.dta"))
    ipeds = ipeds[['unitid', 'instnm']].drop_duplicates().dropna().reset_index(drop=True)

    df_queries = ipeds
    df_candidates = bschool

    query_id = 'unitid'
    candidate_id = 'id'
    query_column = 'instnm'
    candidate_column = 'School'

    df_queries[[query_id]] = df_queries[[query_id]].astype(int)
    df_candidates[[candidate_id]] = df_candidates[[candidate_id]].astype(int)
    df_queries[[query_column]] = df_queries[[query_column]].astype(str)
    df_candidates[[candidate_column]] = df_candidates[[candidate_column]].astype(str)
    # create entity and direction
    df_queries['entity'] = determine_entity(df_queries,query_column, entity_list)
    df_candidates['entity'] = determine_entity(df_candidates,candidate_column, entity_list)
    df_queries['direction'] = determine_entity(df_queries,query_column, directions_list)
    df_candidates['direction'] = determine_entity(df_candidates,candidate_column, directions_list)
    
    keywords_list = ['of', 'the', 'and', 'at','lnc','main campus'] + entity_list
    keywords_dict = {
        'technology': 'tech', 'technical': 'tech', 'state': 'stat',
        'community': 'comm', 'christian': 'christ', 'north': 'abcde',
        'south': 'fghij', 'east': 'klmno', 'west': 'pqrst', 'saint': 'st',
        'aampm': 'a m', 'aampt': 'a t'
    }
    df_queries[f'{query_column}_clean'] = clean_column(df_queries[query_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    df_candidates[f'{candidate_column}_clean'] = clean_column(df_candidates[candidate_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    
    
    # ngram overlap
    # ngram_overlap = ngrams_overlap(df_queries=df_queries,df_candidates=df_candidates, 
    #                             query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean',
    #                             n=4)

    indexer = recordlinkage.Index()
    indexer.block(left_on=['entity', 'direction'], right_on=['entity', 'direction'])
    candidate_links = indexer.index(df_queries, df_candidates)
    
    indexer = NgramsOverlap(query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean', n=4)
    ngram_overlap = indexer.index(df_queries, df_candidates)
    
    candidate_links = candidate_links.intersection(ngram_overlap)
    compare = recordlinkage.Compare(n_jobs=max(os.cpu_count()*0.9, 1))
    compare.string(f"{query_column}_clean", f"{candidate_column}_clean", method='levenshtein', label='name_similarity')
    features = compare.compute(candidate_links, df_queries, df_candidates).reset_index()
    features = features.rename(columns={'level_0': 'query_index', 'level_1': 'candidate_index'})
    features = features.sort_values('name_similarity', ascending=False).drop_duplicates(subset='query_index').drop_duplicates(subset='candidate_index')
    df = pd.merge(df_queries, features, how='inner', left_on=df_queries.index,right_on='query_index').merge(df_candidates, how='inner', left_on='candidate_index',right_on=df_candidates.index)
    df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id).drop_duplicates(subset=candidate_id)
    df = df[df['name_similarity']>0.85]
    ipeds = ipeds.merge(df[['unitid','bschool']], how='left', on='unitid')
    ipeds.to_csv(os.path.join(temp_data_path, "ipeds_with_US_bschool.csv"))
    print("bring in bschool into ipeds!",flush=True)
















    """
    round2
    match revelio with ipeds
    using: bschool name
    """
    revelio = pd.read_csv(os.path.join(raw_data_path, "mba_students_education.csv.gz"),usecols=['university_name','degree'],compression='gzip').drop_duplicates().dropna().reset_index(drop=True)
    revelio = revelio[revelio['degree']=='MBA']
    revelio = revelio.drop_duplicates().dropna().reset_index(drop=True).drop(columns=['degree'],axis=1)
    revelio = revelio.merge(revelio_uni_lookup, how='left', on='university_name')
    
    ipeds = pd.read_csv(os.path.join(temp_data_path, "ipeds_with_US_bschool.csv"))
    ipeds = ipeds[['unitid', 'bschool']].drop_duplicates().dropna().reset_index(drop=True)
    
    
    
    query_id = 'university_id'
    candidate_id = 'unitid'
    query_column = 'university_name'
    candidate_column = 'bschool'
    
    df_round1 = pd.read_csv(os.path.join(temp_data_path,f'{query_id}_to_{candidate_id}_round1_match.csv.gz'),compression='gzip')
    
    df_queries = revelio[['university_id','university_name']]
    df_candidates = ipeds[['unitid', 'bschool']]
    df_queries = df_queries[~df_queries.index.isin(set(df_round1[query_id]))]
    df_candidates = df_candidates[~df_candidates.index.isin(set(df_round1[candidate_id]))]

    """
    clean data
    """
    # unify data type
    df_queries[[query_id]] = df_queries[[query_id]].astype(int)
    df_candidates[[candidate_id]] = df_candidates[[candidate_id]].astype(int)
    df_queries[[query_column]] = df_queries[[query_column]].astype(str)
    df_candidates[[candidate_column]] = df_candidates[[candidate_column]].astype(str)

    
    # create entity and direction
    df_queries['entity'] = determine_entity(df_queries,query_column, entity_list)
    df_candidates['entity'] = determine_entity(df_candidates,candidate_column, entity_list)
    df_queries['direction'] = determine_entity(df_queries,query_column, directions_list)
    df_candidates['direction'] = determine_entity(df_candidates,candidate_column, directions_list)
    
    keywords_list = ['of', 'the', 'and', 'at','lnc','graduate'] + entity_list
    keywords_dict = {
        'technology': 'tech',     'technical': 'tech',     'state': 'stat',     'community': 'comm',
        'christian': 'christ',       'saint': 'st',           'aampm': 'a m',      'aampt': 'a t',
        'business':'busi', 'management':'mgmt', 'economics':'econ'
    }
    df_queries[f'{query_column}_clean'] = clean_column(df_queries[query_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    df_candidates[f'{candidate_column}_clean'] = clean_column(df_candidates[candidate_column], keywords_list=keywords_list, keywords_dict=keywords_dict)
    
    
    """
    only use name similarity
    """
    indexer = recordlinkage.Index()
    indexer.block(left_on=['entity', 'direction'], right_on=['entity', 'direction'])
    candidate_links = indexer.index(df_queries, df_candidates)
    
    indexer = NgramsOverlap(query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean', n=4)
    ngram_overlap = indexer.index(df_queries, df_candidates)
    
    candidate_links = candidate_links.intersection(ngram_overlap)
    compare = recordlinkage.Compare(n_jobs=max(os.cpu_count()*0.9, 1))
    compare.string(f"{query_column}_clean", f"{candidate_column}_clean", method='levenshtein', label='name_similarity')
    features = compare.compute(candidate_links, df_queries, df_candidates).reset_index()
    features = features.rename(columns={'level_0': 'query_index', 'level_1': 'candidate_index'})
    features = features.sort_values('name_similarity', ascending=False).drop_duplicates(subset='query_index').drop_duplicates(subset='candidate_index')
    df = pd.merge(df_queries, features, how='inner', left_on=df_queries.index,right_on='query_index').merge(df_candidates, how='inner', left_on='candidate_index',right_on=df_candidates.index)
    df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id).drop_duplicates(subset=candidate_id)
    ## multithread version
    # chunk_size = max(1000, len(df_queries) // os.cpu_count())
    # num_chunks = (len(df_queries) + chunk_size - 1) // chunk_size
    # df_queries_chunks = np.array_split(df_queries, num_chunks)

    # def process_chunk(df_queries_chunk):
    #     global df_candidates,query_id,candidate_id,query_column,candidate_column
    #     indexer = NgramsOverlap(query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean', n=4)
    #     ngram_overlap = indexer.index(df_queries, df_candidates)
    #     # ngram_overlap = ngrams_overlap(df_queries=df_queries_chunk,df_candidates=df_candidates, 
    #     #                             query_column=f'{query_column}_clean', candidate_column=f'{candidate_column}_clean',
    #     #                             n=5)
    #     # block 2: with the same entity and direction
    #     indexer = recordlinkage.Index()
    #     indexer.block(left_on=['entity', 'direction'], right_on=['entity', 'direction'])
    #     candidate_links = indexer.index(df_queries_chunk, df_candidates)
    #     # get intersection
    #     candidate_links = candidate_links.intersection(ngram_overlap)
    #     compare = recordlinkage.Compare(n_jobs=max(os.cpu_count()*0.9, 1))
    #     compare.string(f"{query_column}_clean", f"{candidate_column}_clean", method='levenshtein', label='name_similarity')
    #     features = compare.compute(candidate_links, df_queries, df_candidates).reset_index()
    #     features = features.rename(columns={'level_0': 'query_index', 'level_1': 'candidate_index'})
    #     features = features.sort_values('name_similarity', ascending=False).drop_duplicates(subset='query_index')
    #     df = pd.merge(df_queries_chunk, features, how='inner', left_on=df_queries_chunk.index,right_on='query_index').merge(df_candidates, how='inner', left_on='candidate_index',right_on=df_candidates.index)
    #     df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id)
    #     return df

    # with tqdm_joblib(tqdm(desc="Processing chunks", total=len(df_queries_chunks))):
    #     df_list = Parallel(n_jobs=-1)(
    #         delayed(process_chunk)(chunk) for chunk in df_queries_chunks
    #     )
    
    # df = pd.concat(df_list)
    # df = df.sort_values('name_similarity', ascending=False).drop_duplicates(subset=query_id)

    # get round2 match
    df_round2 = df[df['name_similarity']>0.88][[query_id,candidate_id,query_column,candidate_column,'name_similarity']]
    df_round2.to_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_round2_match.csv.gz'), 
                index=False,
                compression='gzip')
    print("round2 done!",flush=True)
















    """
    concat round1 and round2
    """
    df_round1 = pd.read_csv(os.path.join(temp_data_path,f'{query_id}_to_{candidate_id}_round1_match.csv.gz'),compression='gzip')
    df_round2 = pd.read_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_round2_match.csv.gz'),compression='gzip')

    df = pd.concat([df_round1, df_round2], axis=0).drop_duplicates().reset_index(drop=True)
    
    ipeds = pd.read_csv(os.path.join(temp_data_path, "ipeds_with_US_bschool.csv"))
    df['merge_uni'] = df['instnm'].notnull().astype(int)
    df['merge_bschool'] = df['bschool'].notnull().astype(int)
    
    df = df.drop(columns=['instnm','bschool'],axis=1)
    df = df.merge(ipeds[['unitid', 'bschool', 'instnm']], how='left', on='unitid')

    reorder_list = [
        'university_id',
        'unitid',
        'university_name',
        'instnm',
        'bschool',
        'merge_uni',
        'merge_bschool',
        'name_similarity',
    ]
    df = df[reorder_list]
    df.to_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_matched.csv.gz'),
                index=False,  
                compression='gzip')
    print("ALL done!",flush=True)
    df= pd.read_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_matched.csv.gz'))



    """
    get_user_id_with_US_mba_degree
    """
    
    query_id = 'university_id'
    candidate_id = 'unitid'
    query_column = 'university_name'
    
    revelio = pd.read_csv(os.path.join(raw_data_path, "mba_students_education.csv.gz"),usecols=['user_id','university_name','degree'],compression='gzip').drop_duplicates()
    revelio = revelio[revelio['degree']=='MBA']
    revelio = revelio.drop_duplicates().dropna().reset_index(drop=True).drop(columns=['degree'],axis=1)
    revelio_matched_unis =  set(pd.read_csv(os.path.join(temp_data_path, f'{query_id}_to_{candidate_id}_matched.csv.gz'), 
                                    usecols=['university_name'],
                                    compression='gzip')['university_name'])
    
    revelio = revelio[revelio['university_name'].isin(revelio_matched_unis)]
    user_id_with_US_mba_degree = revelio[['user_id','university_name']].drop_duplicates().reset_index(drop=True).rename(columns={'university_name':'MBA_university_name'})

    user_id_with_US_mba_degree.to_csv(
        os.path.join(temp_data_path, 'user_id_with_US_MBA_degree.csv.gz'),
        compression='gzip',
        index=False)
    
    user_id_with_US_mba_degree.to_csv(
        os.path.join(temp_data_path, 'user_id_with_US_MBA_degree.csv'),
        index=False)

