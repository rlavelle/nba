# nba

TODO: 

Betting structure: 
    - probability score for prop bets (similar to how ml is done)
        - in P(model points | vegas points)?
    - betting system (unit assignment)
        - edge is most likely contained within the diff from preds to vegas

Modeling:
    - spread model needs revamping, does not work...
    - pre season data should be used with some decay weighting
    - is win streak a worthy predictor? 

Refactoring:
    - project structure needs a rework
    - job scripts need to become more abstract and wrapped in classes
        - abstract pipeline class, with all jobs in their own folder
        - should a pipeline class and a job file be separate?
    - theres a lot of similar patterns with retries ect. decorators?
    - error codes within the API and the jobs need to be reworked
    - do games need to be decoupled from dates when writing jsons to the db?
        - this is causing some incomplete games to make other games on the same day be dropped
    - feature engineering needs to be overworked
        - increase runtime by only creating needed features
    - need to fix ?leak? bug on modeling from the cache -- why is this happening?
        - diff results occur if you build the features, vs subset the cached feature set
