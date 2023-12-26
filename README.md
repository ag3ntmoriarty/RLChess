# RLChessAgent
Make a at first a chess engine with lil previous knowledge
Train a similar 0 knowldge chess engine 


* Establish board
    * Pieces:
    * King
    * Queen
    * Rook
    * Bishop
    * Knight
    * Pawn

TO DO:
Value Net:
Need to design value net ie. the one to judge and reduce the search tree 
thinking of utilising resnet block for evaluation
Approaches:
CNNs with ReLU Activation

Thinking of combining the policy and value net in 1 model can do that in pytorch but need to shorten the model to get better performance.

Policy Net:
override LegalMoveGenerator to ge list

Need to find way to get a net where the number of output changes wrt the legal moves on board the output size used is 8x8x73 ()

AlphaGo- Uses a policy net(cnv2d and relu) to predict human actions in game using random state action pair if humans (approach 1)

Used RL Policy training with approach 1 model and weights to do self play and improve play prediction. weight updated based on the policy gradient * outcome of the game

implementd value and policy net in the same model in a reconstructin of the alphago paper chess deep rl on github

store multiple model in one pth file.
(Need to check other agents)
The input size is variable at this point need to refine to either 64xn or 8x8xn



* Make a search tree for games possible combinations
* Reduce search tree with neural net (Value Net).. V = f(board)
* RL Agent (Policy Net) to evaluate the net and take the best possible path
* Self play to improve policy net and the value net is trained on classic SL approach