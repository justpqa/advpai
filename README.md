### File structure

| File | Description |
| --- | --- |
| `chroma_db/` | Folder for Chroma DB for RAG |
| `model` | Folder for models, each subfolder represent a model |
| `other` | Folder for other experimental files that is not used by current pipeline |
| `papers/` | Folder for set of papers used for exploring different approach, mostly not used more |
| `pred_tables/` | Set of harmonized tables from test_papers folder without text col, to be distinguished with pred_tables_with_text_col |
| `pred_tables_with_text_col/` | Given a harmonized table from pred_tables_with_text_col, we harmonize the text col from paper and append onto that and finally save here |
| `referencing_cols` | Folder for descriptions related to different columns (meanings, examples, etc.) |
| `tables` | Intermediate tables created in the process before get to pred_tables |
| `test` | Folder for evaluation/test files, mostly to evaluate on text columns since they are hard to do |
| `test_logs` | Folder for test logs for testing accuracy of different text columns |
| `test_papers` | Folder for papers used to test the harmonization pipeline |
| `.env` | Config file |
| `.gitignore` | Including files that are not tracked by github |
| `advp_paper_searching.py` | Python script for searching new Alzheimer's Disease-related GWAS papers => output are new_gwas_ad_paper*.csv/.xlsx files |
| `advp_paper_searching_experiment.ipynb` | Notebook for experimenting with paper search, including exploring title and abstract from harmonized paper => find new set of title and abstract terms to add into filter for advp_paper_searching.py (terms are at most 4 words) => similar to advp_paper_tiab_analysis.iypnb |
| `new_paper/new_gwas_ad_paper*.csv/.xlsx` | Files for output of different iterations of advp_paper_searching.py: new_gwas_ad_paper: original set of keywords from Fanny's work, _extended: adding some more terms from first advp_paper_tiab_analysis, _extended_1: same as previous, but it is an extra loop, _extended_converged: the converged list of paper after 2 iteration |
| `count_new_paper/count_new_paper_by_disease*.csv/.xlsx` | Count the number of papers that in each different disease category through different paper searching iteration: count_new_paper_by_disease: original set of keywords from Fanny's work, _extended: adding some more terms from first advp_paper_tiab_analysis, _extended_1: same as previous, but it is an extra loop, _extended_converged: the converged list of paper after 2 iteration |
| `tiab_terms_counter/` | After advp_paper-searching_experiments run, we will have a list of different n-grams in different title and abstract parts of ADVP paper along with their count, across different iteration of harmonizing new paperss |
| `approach.txt` | Notes on approach of my model, not used anymore |
| `notes_on_columns_to_match.txt` | Notes on specific caveat of matching certain columsn, not used anymore |
| `cohort_keywords.json` | Dict mapping 1 universal name for a cohort to a list of different ways that represent it, used to be used by information retriever to extract cohort info, might not need them |
| `advp_table_extraction.py` | Code for engine for extracting different tables from paper, can test some of that code in advp_table_extraction_experiment.iypnb |
| `advp_formatting_engine.py` | Code for engine for formattting different tables into good format, can test some of that code in advp_table_extraction_experiment.iypnb |
| `advp_information_retriever.py` | Code for the information retriever from a paper: extract different info that is align to a certain information category => used later for harmonizing text columns (using RAG: find possible chunks, extract possible info from LLM) |
| `advp_table_extraction_experiment.ipynb` | Notebook for experimenting different methods to extract tables from ADVP papers => harmonize them |
| `id_to_table_id_lst.json` | Dictionary of paper id -> list of table id that Pubmed central used in order to extract table later using PMC API |
| `ingest_papers.py` | Old code for ingesting papers into Chroma DB before harmonizing, might not need it in current versions of info retriever |
| `model_servers_init.sh` | Bash script to initialize API servers to serve model locally so that different python scripts can call and use these models |
| `test/conftest.py` | Test config, needed to initialize pytest script |
| `test/extract_test_tables_advp1.py` | Script for extracting test tables correspond to sample set of papers we test from ADVP1 from the full ADVP1 ground truth of all ADVP1 papers |
| `test/test_advp1.py` | Set of test/evaluation functions for evaluating harmonization results for different col from ADVP1 papers |
| `test/term_mapping_dict.json` | Set of ground truth terms in test -> set of possible terms that have same meaning, created with the help of Claude |
| `test/term_validation_results.json` | Paper => information on harmonization success at each different text col, saved from test_advp1.py |
| `test_matching_dict.json` | Dict of paper -> dict of {matched term: score, infinity for exactly similar term} for the harmonization of col that can be found directly from tables|
| `util.py `| Other utils function needed by searching or extraction or harmonization |
