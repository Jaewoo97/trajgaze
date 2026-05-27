Gaze-hand Trajectory Merging for Efficient
Egocentric Video Understanding
AnonymousAuthor(s)
Affiliation
Address
email
Abstract
Egocentricvideounderstandingwithmultimodallargelanguagemodelsislimited
1
bythehighvisual-tokencostoflongfirst-personvideos. Generictokencompres-
2
sionmethodsoftendiscardtask-criticalregionsbecausetheyfailtocapturehuman
3
intent. Inegocentricsettings,however,gazeanticipatesupcomingactionswhile
4
hands realize them through physical interaction, providing a strong behavioral
5
proxyforthecamerawearer’sintent. WeproposeTrajGazeMerge,atrajectory-
6
guided token merging framework that leverages gaze and hand trajectories to
7
preserve behaviorally salient visual tokens under aggressive compression. Our
8
methodpredictsper-framepatchimportancebyjointlymodelinggaze,left-hand,
9
right-hand,andgaze-handinteractionsignals,thenalignsthesescoreswithvideo
10
MLLM tokens for score-weighted bipartite merging. This allows the language
11
modeltoprocessamuchshortervisualsequencewhileretainingtokenslikelyto
12
encodeuserintent. Trainingconsistsoftrajectorypretrainingwithtrajectoryand
13
patch-importanceobjectives,followedbyjointfine-tuningofthetrajectoryencoder
14
andLoRA-adaptedVLMthroughthedifferentiablescore-weightedaveragingstep
15
ofthemerge. OnStreamGaze,trainedonEgoExoLearnandHoloAssistandeval-
16
uatedonthedisjointEGTEAportion, TrajGazeMergeachieves68.44%overall
17
accuracyatonly10%visual-tokenretention,outperforminguniformsubsampling,
18
attention-basedpruning,andcontent-basedmergingunderthesametokenbudget.
19
1 Introduction
20
Egocentricvideohasemergedasakeyinputmodalityforhuman-centricapplications, including
21
augmentedreality,assistiverobotics,andbehavioralanalysis[1–5].Recentmultimodallargelanguage
22
models(MLLMs)[6–11]havemadevisualquestionanswering(VQA)increasinglypractical[12–
23
15], yet egocentric footage poses distinctive challenges, including frequent viewpoint changes,
24
dense hand-object interactions, and long-horizon temporal dependencies. These challenges are
25
sharpenedbythevisual-tokenbottleneckofvideoMLLMs[16]: egocentricclipsroutinelyspantens
26
ofminutes[3,13], whilehand-objectinteractionregionsoftenrequirehighspatialresolutionfor
27
fine-grainedreasoning[17,18]. Forinstance,fast-movinghandsorsmallmanipulatedobjectsoften
28
blendintothevisualbackground,causinggenericcompressionalgorithmstooverlooktheseregions
29
despitetheircriticalimportancetothetask. Naivetemporalsubsampling,acommonfallbackunder
30
tighttokenbudgets,canthereforediscardspatialandtemporalevidencethategocentrictasksdepend
31
on.
32
Unlikethird-personvideo,egocentricfootageisnaturallyaccompaniedbybehavioralsignalstied
33
tothecamerawearer,mostnotablygazeandhandtrajectories[18–22]. Naturalisticstudiesshow
34
thatgazeineverydaytasksisstronglytask-drivenandoftenanticipatesupcomingmanualactions
35
inajust-in-timemanner[23,24],incontrasttobottom-upsaliencyaccountsofattention[25,26].
36
Theresultingeye-handcoupling,wheregazeindicateswheretheagentislikelytoactandthehand
37
Submittedto40thConferenceonNeuralInformationProcessingSystems(NeurIPS2026).Donotdistribute.

realizesthatintention[22],providesausefulbehavioralproxyforuserintent. Thesesignals,however,
38
areimperfect: gazetrackingcanbemissingornoisy,andhandsfrequentlyleavethefieldofview.
39
A compression method that conditions on them must therefore use gaze and hand trajectories as
40
informativebutfalliblecues,ratherthanashardspatialconstraints.Wearguethatthetokenbottleneck
41
inegocentricvideoshouldthereforebeaddressednotonlythroughgenericvisualcompression,but
42
alsobyconditioningcompressiononbehavioralcuesalreadypresentinthestream.
43
Existingworkonefficientvideounderstandinghasexploredframeselection, tokenpruning, and
44
tokenmerging[27–43]. Thesemethodsarelargelydrivenbygenericvisualstatisticsorquery-visual
45
relevance,whichmayfailtoprioritizeegocentricevidenceassociatedwithhand-objectinteractionand
46
task-drivenfixationpatternsunderaggressivecompression. Recentgaze-awareegocentricMLLMs
47
showthatgazeprovidesusefulintentcues[15,44],buttheydonottargetseveretokenreduction.
48
Existing token-reduction frameworks have also rarely used gaze and hand trajectories jointly as
49
behavioralconditioningsignalsforpatch-levelimportanceprediction.
50
We present TrajGazeMerge and evaluate it on StreamGaze [45], a streaming egocentric VQA
51
benchmarkcomposedofEGTEA[18],EgoExoLearn[46],andHoloAssist[5]. Unlikesparse-frame
52
gaze-MLLMbenchmarkssuchasEgoGazeVQA[15],StreamGazeprovideslongegocentricstreams
53
withsynchronizedgazeandbimanualhandtracking,makingitwell-suitedforstudyingtrajectory-
54
conditioned token compression under non-trivial visual-token budgets. TrajGazeMerge predicts
55
patch-levelimportancefromgaze,hand,andgaze-handinteractiontrajectories,andusesthesescores
56
toguidescore-weightedtokenmergingbeforevisualtokensenterthelanguagemodel. Followingthe
57
StreamGazesourcecomposition,wetrainontheEgoExoLearnandHoloAssistportionsandevaluate
58
onthedisjointEGTEAportion,allowingustoexaminewhetherbehavior-conditionedcompression
59
remainseffectiveacrossvisualsources. Theevaluationspanseightmultiple-choicetaskcategories
60
thatcoverpastandpresentreasoningaboutgazeandscenecontext.1 Atatokenretentionratioof10%,
61
TrajGazeMergeachieves68.44%overallaccuracyonEGTEA,outperforminguniformsubsampling,
62
attention-basedpruning,andcontent-basedmergingatthesamecompressionbudget. Thesegainsare
63
especiallypronouncedontasksdemandinglong-horizonvisualmemory(pastscenerecall,+13.5pp
64
overuniformsubsampling)andfine-grainedobjectdiscrimination(presentobjectidentificationhard),
65
wheregenericcompressionmaydiscardtask-criticalevidence.
66
Ourcontributionsareasfollows:
67
• WeproposeTrajGazeMerge,anoveltoken-mergingframeworkthatleveragesatrajectory-
68
conditionedpatch-importancepredictor. Byjointlymodelinggaze,handtrajectories,and
69
eye-hand interactions, the predictor uses trajectory context as a behavioral conditioning
70
signal to estimate token importance for egocentric token merging, rather than directly
71
enforcinggazeorhandretention.
72
• Weintroduceatwo-stagetrainingframeworkfortrajectory-guidedtokenmerging. Stage1
73
pretrainsthetrajectoryencoderwithfourcomplementaryobjectives: trajectoryprediction,
74
patch-scoreprediction,spatialgrounding,andtrajectory-drivenscoredistillation. Stage2
75
jointlyfine-tunesLoRA-adaptedLLMweightsandthetrajectoryencoderthroughadiffer-
76
entiablescore-weightedbipartitetoken-mergingoperation,withoutrequiringanexternal
77
teachermodel.
78
• We provide an empirical validation of behavior-conditioned token compression on
79
StreamGazeunderanaggressivetokenbudget,achieving68.44%overallaccuracyonthe
80
disjointEGTEAportionat10%tokenretentionandoutperforminguniformsubsampling,
81
attention-basedpruning,andcontent-basedmergingatequalbudgets.
82
2 RelatedWork
83
2.1 EfficientVideoUnderstandingwithMLLMs
84
AdaptingMLLMstolongvideohasdrivenasurgeofworkonvisual-tokencompression,sincetempo-
85
rallyextendedsequencesquicklyexceedthecontextwindow[36,37,31,32,34,35]. Representative
86
systems address this bottleneck through unified video-language representations, per-frame token
87
1Wefocusonmultiple-choiceVQA;twoproactivetasksintheoriginalbenchmarkareexcludedastheyareformulatedas
free-formalerts.
2

compression,hierarchicalcompression,orspatiotemporaltokenreduction[11,8,32,31,34,35]. A
88
complementarylinereducesthenumberofframesbeforeorduringinference: adaptivekeyframe
89
samplingselectsinformativeframesfromlongvideos[33], Q-Frame[38]performsquery-aware
90
selectionwithmulti-resolutionadaptation,andMDP3[39]formulateslist-wiseselectionthatbalances
91
queryrelevance,diversity,andtemporalsequentiality. Thesemethodsshowthatuniformsampling
92
issuboptimalundertightvisualbudgets,buttheyoperateatframegranularityanddonotexploit
93
egocentricbehavioralsignals.
94
Patch-leveltokenreductionprovidesanotherroutetoefficiency. Earlymethodsprunelow-importance
95
tokens [27, 30, 28], while recent VLM-specific approaches compress or remove visual tokens
96
duringmultimodalinference[40,36,41–43]. Tokenmergingavoidsoutrightremovalbycombining
97
redundant tokens, as in ToMe [29], which greedily matches and averages similar token pairs via
98
bipartite soft matching. TrajGazeMerge inherits this bipartite merging structure but changes the
99
selectioncriterion: ratherthanrelyingsolelyonvisualsimilarity,attentionscores,orquery-frame
100
relevance, it derives patch-importance scores from egocentric gaze and hand behavior. This lets
101
mergingdecisionsreflecthumanbehavioralcontextbeforethelanguagemodelprocessesthevisual
102
tokens. Tothebestofourknowledge,TrajGazeMergeisthefirsttodirectlyintegratecontinuous,joint
103
gaze-handkinematicsignalsintoadifferentiabletokenmergingpipelineforMLLMs. Thisapproach
104
is,inprinciple,combinablewithframe-levelselectionmethodsthatoperateatadifferentgranularity.
105
2.2 EgocentricVideoUnderstanding
106
Egocentricvideounderstandinghasbeenadvancedbylarge-scalefirst-persondatasets,including
107
EPIC-Kitchens[1,2],Ego4D[3],EgoExo4D[4],EgoExoLearn[46],EGTEA[18,47],andHoloAs-
108
sist[5].Thesedatasetscoverfine-grainedactivities,episodicmemory,cross-viewprocedurallearning,
109
gaze-annotatedactionrecognition,andinteractiveassistance. Buildingonthem,vision-languagepre-
110
trainingandmultimodalreasoninghavebecomecentralparadigms.EgoVLP[48]andEgoVLPv2[49]
111
learnegocentricvideo-languagerepresentations, whileEgoTaskQA[12], EgoSchema[13], MM-
112
Ego [14], EgoTextVQA [50], HiERO [51], and Ego-R1 [52] study task-oriented, long-form, or
113
reasoning-orientedegocentricVideoQA.Toaddresstheunderexploredareaofbehavior-guidedlong-
114
horizonreasoning,weevaluateonStreamGaze[45],usingitseightmultiple-choicetasksthatcover
115
pastandpresentreasoning,whileexcludingfree-formproactivealerts.
116
2.3 GazeandHandAttentioninEgocentricVideo
117
Gaze has long been recognized as a privileged signal in egocentric settings. Cognitive studies
118
showthateyemovementsinnaturalbehavioraretightlycoupledtotaskgoalsandoftenanticipate
119
manualactionsinajust-in-timemanner[23,24],whilecomputationalmodelsofattentioninitially
120
emphasizedbottom-upsaliency[25,26]. Egocentricgazepredictionmethodsfurthershowthatgaze
121
isstructuredbytaskcontextandfutureactions[19,20]. Gazeandhandbehaviorhavealsobeen
122
usedforactionunderstanding,affordancediscovery,andinteractionprediction: gazecorrelateswith
123
manipulatedobjects[18,47],revealsaffordancehotspots[21],andcomplementshand-contactand
124
hand-motioncuesforinteractionunderstanding[17,22]. Together,theseworkssupporttheviewthat
125
gazeandhandsprovidecomplementarybehavioralevidence,wheregazeindicateswherethewearer
126
islikelytoattendoract,whilehandsrealizethatintentphysically.
127
Morerecently,gazehasbeenintegratedintoegocentricMLLMandVQAsettings.EgoGazeVQA[15]
128
benchmarksgaze-guidedegocentricvideointentunderstanding,andGaze-VLM[44]bridgesgaze
129
andVLMsthroughattentionregularization. Theseworksestablishgazeasaneffectivesignalfor
130
egocentricMLLMs,butdonottargetseverevisual-tokenreduction. Incontrast,TrajGazeMergeis
131
specificallydesignedforaggressivetokenreduction. Ittreatsthecontinuousspatiotemporalflowof
132
recordedgazeandhandtrajectoriesasdirectinputstoalearnedencoder,producingnoise-robust,
133
patch-level importance scores. This directly connects continuous, egocentric behavioral signals
134
toefficientvideo-languagemodeling. WeaccordinglyevaluateonStreamGaze[45], whoselong
135
egocentricstreams,withsynchronizedgazeandbimanualhandtracking,arewell-suitedforstudying
136
trajectory-conditionedtokencompression.Thiscontrastswithsparse-framegaze-MLLMbenchmarks
137
suchasEgoGazeVQA[15],whichprimarilyevaluategazecuesoverasmallsetofsampledframes.
138
3

|     |     |     |     |     |     | Trajectory ReasoningSec. 3.2.1 |     |     |     | Trajectory-Visual FusionSec. 3.2.3 |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ---------------------------------- | --- | --- |
Egocentric Video
|     |     |     |     |     | Gaze |     |     |     |     |     | Cross Attention |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --------------- | --- |
...
Left Hand
K, V
Right Hand
|     |     |     |     |     | Interaction |     |     |     |     | DINOv2 |     |     |
| --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | ------ | --- | --- |
Behavioral Trajectories
Intra-frame Fusion
Gaze
|     | Left Hand  |     |     |     |     | Temporal Transformer |     |     |     |     |              |     |
| --- | ---------- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | ------------ | --- |
|     | Right Hand |     |     |     |     |                      |     |     |     |     | Patch Scores |     |
Interaction
Trajectory Context Tokens
Merge to Nearest Patch
|     |      | Input Trajectories |     | Intra-fra | m e Fusion | Tem p ora.l. T.ra | n s form er |     |     |     |     |     |
| --- | ---- | ------------------ | --- | --------- | ---------- | ----------------- | ----------- | --- | --- | --- | --- | --- |
|     | Gaze |                    |     |           | . ..       | t- K              | t - 1 t     |     |     |     |     |     |
Left Hand
Ratio r
Right Hand
Interaction
|     |     |     | Interaction Token Features |     |     |     |     | What is the person gazing at? |     |     |     |     |
| --- | --- | --- | -------------------------- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- |
Distance Convergence Lead-Lag Relative Velocity Relative Direction Compressed Tokens
|     | D(g, h) |     |     |      | (Δv) |     | (Δθ) |     |     |            |      |     |
| --- | ------- | --- | --- | ---- | ---- | --- | ---- | --- | --- | ---------- | ---- | --- |
|     |         |     |     |      |      |     |      |     |     | Qwen2.5-VL | LoRA |     |
|     |         |     |     | Time |      |     | θ    |     |     |            |      |     |
Figure1: OverviewofTrajGazeMerge. Ourframeworkleveragesgaze,hand,andeye–handinterac-
tiontrajectoriesasaproxyforuserintent,guidingtokencompressiontoretainthemostinformative
visualcuesforefficientegocentricvideounderstanding.
3 Method
139
| 140 | 3.1 ProblemFormulation |     |     |     |     |     |     |     |     |     |     |     |
| --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈R2,left/right
141 LetV ={I }T beaT-frameegocentricvideoclipwhereeachframecarriesgazeg
|     |                   | t t=1 |     |                                           |     |     |     |     |     |          | t              |     |
| --- | ----------------- | ----- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | -------- | -------------- | --- |
|     | handpositionshL/R |       |     | R2,velocitiesh˙L/R,andvisibilityflagsmL/R |     |     |     |     |     |          |                |     |
| 142 |                   |       | ∈   |                                           |     |     |     |     |     | ∈ {0,1}. | Givenaquestion |     |
|     |                   |       | t   |                                           |     | t   |     |     |     | t        |                |     |
143 q and K candidates O = {o }K , the task is to predict k∗. A frozen visual backbone encodes
|     |     |     |     | k   | k=1 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
eachframeintoP=196patchtokens,yieldingN =T·P tokens;a128-frameclipexceeds25,000
144
tokens. Ourgoalistoselect⌊ρN⌋tokens(ρ=0.10)guidedbybehavioralsignals(G,HL,HR)and
145
q,withoutaccessingVLMinternalsatselectiontime.
146
|     | 3.2 TrajGazeMerge |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
147
|     |     |     |     |     |     |     | (cid:0) |     |     | (cid:1) |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ------- | --- | --- |
TheTrajGazeMergemapsthetrajectoryprefix G ,HL ,HR ,anoptionalqueryq,andT
| 148 |     |     |     |     |     |     | 1:Tp | 1:Tp | 1:Tp |     |     | p   |
| --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | --- | --- | --- |
videoframestoaper-frame,per-patchimportancescorematrixS∈[0,1]Tp×P. Figure1illustrates
149
thefullpipeline.
150
151 Per-modalitytokenization. Foreachpastframet,weprojecteachinputstreamtoad=128token:
|     |     | zg     | (cid:0) |          | (cid:1)  |         |     | zL     | (cid:0) | [hL;        | h˙L; mL] (cid:1) |     |
| --- | --- | ------ | ------- | -------- | -------- | ------- | --- | ------ | ------- | ----------- | ---------------- | --- |
|     |     | =LN    | W       | g [g t ; | g˙ t ] , |         |     | =LN    | W       | L           | ,                |     |
|     |     | t      |         |          |          |         |     | t      |         | t           | t t              |     |
|     |     |        | (cid:0) |          |          | (cid:1) |     |        | (cid:0) |             | (cid:1)          |     |
|     |     | zR =LN | W       | [hR;     | h˙R; mR] | ,       |     | zϕ =LN | W       | ϕ(g ,hL,hR) | ,                | (1) |
|     |     | t      |         | R t      | t        | t       |     | t      |         | ϕ t         | t t              |     |
where ϕ ∈ R12 encodes pairwise gaze–hand displacement, relative velocity, convergence, and
152
lead–lag. Missing detections are replaced by a learnable missing embedding, yielding z =
| 153 |          |               |     |     |     |     |     |     |     |     |     | t   |
| --- | -------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | [zg, zL, | zR, zϕ]∈R4×d. |     |     |     |     |     |     |     |     |     |     |
154
|     | t t | t t |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Trajectorytransformer. A2-layer,4-headtransformerfirstfusesthefourwithin-frametokensper
155
frameintoz˜ =TF (z ). TheresultisprojectedtoD=256,flattenedtolengthT ·4withsinusoidal
| 156 |     | t   | L1  | t   |     |     |     |     |     |     | p   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
157 positionalencodings,andprocessedbya6-layer,8-headtransformer:
|     |     |     |     | x←TF | (cid:0) | PE(W | [z˜  | ]) (cid:1) ∈R(Tp·4)×D, |     |     |     |     |
| --- | --- | --- | --- | ---- | ------- | ---- | ---- | ---------------------- | --- | --- | --- | --- |
|     |     |     |     |      | L2      | proj | 1:Tp |                        |     |     |     | (2) |
capturinglong-rangegazeanticipationandhand-motionpatterns;reshapedtoX∈RTp×4×D.
158
4

∈Rdq;aFiLMlayer[53]
| 159 | QueryconditioningviaFiLM. |     |     |     | AfrozenCLIPencodermapsqtoe |     |     |     |     |     |     |     |
| --- | ------------------------- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- |
q
modulatesthetrajectorycontext,x˜ =x⊙(1+γ(e ))+β(e ),enablingtask-specificweightingof
| 160 |                     |     |                |     |                          | q   |     | q   |     |     |     |     |
| --- | ------------------- | --- | -------------- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- |
|     | gazevs.handstreams. |     | DuringStage1,e |     | =0reducesFiLMtoidentity. |     |     |     |     |     |     |     |
| 161 |                     |     |                |     | q                        |     |     |     |     |     |     |     |
Visualpatchfeatures. AfrozenDINOv2-ViT/Sbackbone[54]encodesK=16keyframesinto
162
V(1:K) ∈RK×P×Dv (D =384,projectedtoD=256byatrainablelinearlayer),linearlyinterpolated
| 163 |     |     | v   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
toV∈RTp×P×D.
164
Trajectory–visualfusion. Foreachframet,query-conditionedtrajectorytokensX˜ cross-attend
| 165 |                  |     |               |          |          |       |      |          |     |     | t   |     |
| --- | ---------------- | --- | ------------- | -------- | -------- | ----- | ---- | -------- | --- | --- | --- | --- |
|     | tovisualpatchesV |     | withH=8heads: |          |          |       |      |          |     |     |     |     |
| 166 |                  |     | t             |          |          |       |      |          |     |     |     |     |
|     |                  |     |               |          | (cid:32) |       |      | (cid:33) |     |     |     |     |
|     |                  |     |               |          | (X˜      | Wh)(V | Wh)⊤ |          |     |     |     |     |
|     |                  |     |               |          |          | t Q√  | t K  |          |     |     |     |     |
|     |                  |     | Ah            | =softmax |          |       |      | ∈R4×P.   |     |     |     | (3) |
|     |                  |     |               | t        |          | d     |      |          |     |     |     |     |
h
|     | Attendedfeaturesareaddedresidually,Xˆ |           |     |     | =X˜ |     | (cid:0)(cid:76) |     | (cid:1)                  |     |     |     |
| --- | ------------------------------------- | --------- | --- | --- | --- | --- | --------------- | --- | ------------------------ | --- | --- | --- |
|     |                                       |           |     |     |     | +W  | Ah(V            | Wh) | ,yieldingenrichedcontext |     |     |     |
| 167 |                                       |           |     |     | t   | t O | h               | t t | V                        |     |     |     |
|     | Xˆ ∈RTp×4×D.                          | BecauseX˜ |     |     |     |     |                 |     |                          |     |     |     |
168 alreadycarriesglobaltemporalcontextfromEq.(2),spatialattentionat
|     |                                                       |     |     | t   |     | Themeancross-attentionA¯ |     |            |     |                   | (cid:80)  |     |
| --- | ----------------------------------------------------- | --- | --- | --- | --- | ------------------------ | --- | ---------- | --- | ----------------- | --------- | --- |
|     | eachframeisconditionedonthefulltrajectory.            |     |     |     |     |                          |     |            |     | [p]=              | 1 Ah[i,p] |     |
| 169 |                                                       |     |     |     |     |                          |     |            |     | t                 | 4H h,i    | t   |
|     | isrefinedbyalightweightheadintoauxiliaryscoremapsSenc |     |     |     |     |                          |     | ∈[0,1]Tp×P |     |                   |           |     |
| 170 |                                                       |     |     |     |     |                          |     |            |     | usedonlyforStage1 |           |     |
supervision(§3.3).
171
Atinference,adedicatedheadpoolsXˆ
|     | Trajectory-drivenscorehead. |     |     |     |     |     |     |     |     | vialearnedattentionand |     |     |
| --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- |
| 172 |                             |     |     |     |     |     |     |     | t   |                        |     |     |
projectstoper-patchscores:
173
|     |     | (cid:80) |     |     |     |     |     | (cid:0) |     |     | (cid:1) |     |
| --- | --- | -------- | --- | --- | --- | --- | --- | ------- | --- | --- | ------- | --- |
c = αtXˆ [i], αt =softmax(W LN(Xˆ )), S[t,·]=σ W GELU(W LN(c )) ∈[0,1]P.
|     | t   | i i t |     |     | α   | t   |     |     | 2   | 1   | t   |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(4)
DuringStage1,thisheadissupervisedagainsttrajectory-prediction-driventargets,ensuringinference
174
175 scoresreflectgenuinepatchrelevanceratherthanrawsaliency.
TwolightweightdecodersoperateonXˆ
| 176 | Stage-1decoders. |     |     |     |     |     |     | duringStage1onlyandarediscarded |     |     |     |     |
| --- | ---------------- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- |
atinference. AbankofT learnedqueriescross-attendstoflattenedcontextXˆ througha3-layer
| 177 |     |     | f   |     |     |     |     |     |     | flat |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
transformer,predictingallfuturestepsinparallel:thetrajectorydecoderpredictsfuturegazeandhand
178
|     | positionsPˆ | ∈[0,1]Tf×6,andthescoredecoderpredictsper-patchimportanceSˆ |     |     |     |     |     |     |     |     |     |     |
| --- | ----------- | ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈[0,1]Tf×P
| 179 |     | 1:Tf |     |     |     |     |     |     |     |     | 1:Tf |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- |
foreachfutureframe.
180
|     | 3.3 | Training |     |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
181
Trainingproceedsintwostages: Stage1pretrainsthetrajectoryencoderonbehavioraldataalone;
182
Stage2jointlyfine-tunestheencoderwiththeVLMundertasksupervision.
183
Stage1: Trajectorypretraining. ClipsaresubsampledtoT=128frames; thepast/futuresplit
184
isdrawnuniformlyfrom[40%,60%]astemporalaugmentation. Theground-truthpatchrelevance
185
targetis
186
|     |     |     |     | I(p,t)=G(p,t)·H(p,t)·Φ |     |     |     | t ·Ψ t , |     |     |     | (5) |
| --- | --- | --- | --- | ---------------------- | --- | --- | --- | -------- | --- | --- | --- | --- |
187 where G,H are Gaussians centered at gaze/hand positions, and Φ t ,Ψ t are the convergence and
lead–lagscalarsfromϕ,temporallyaveragedoverW=8frames. TheStage1losscombinesfour
188
terms,
189
|     |     |     | L   | =L   | +L        | +L         |     | +L         |     | ,   |     | (6) |
| --- | --- | --- | --- | ---- | --------- | ---------- | --- | ---------- | --- | --- | --- | --- |
|     |     |     | S1  | traj | score-fut | score-past |     | score-traj |     |     |     |     |
whereL ismaskedMSEonfuturepositions,L supervisethedecoderandencoder
| 190 |     | traj |     |     |     | score-fut/past |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
191 scoremapsagainstEq.(5),andthefourthtermsupervisestheinferencescoreheadagainsttrajectory-
192 prediction-driventargets:
4
(cid:88)
Straj[t,p]∝ ω¯ ·A [i,p], L =MSE (cid:0) S[t,·], sg(Straj[t,·]) (cid:1) , (7)
|     |     |     |     | t,i | t   | score-traj |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
i=1
withω¯ themeandecodercross-attentiontowardtrajectorytokeniatframet. Wepretrainon246
| 193 |     | t,i |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
clipsfromEgoExoLearnandHoloAssistfor100epochswithAdamWlr=3×10−4.
194
5

195
Stage 2: Joint fine-tuning with VLM LoRA. The score matrix S ∈ [0,1]Ttraj×P is mapped
toaflatvectors ∈ [0,1]N overallN VLMvisualtokensviabilinearspatialandlineartemporal
196
interpolation. Tokensarethenrankedbyscore;thetopN =⌈ρN⌉becomereceiversRandtherest
197 r
sources S. Following Bolya et al. [29], each source merges into its cosine-nearest receiver with
198
score-weightedaveraging,
199
(cid:80)
j∗(i)=argmax f i ⊤f j , ˆf = s j f j + i:j∗(i)=j s i f i , (8)
∥f ∥∥f ∥ j s + (cid:80) s
j∈R i j j i:j∗(i)=j i
preservingreceiverpositionalidentitieswhilefoldingdiscardedtokensintonearbyrepresentations.
200
WeadaptQwen2.5-VL-7B[55]withLoRA[56]inallattentionprojectionlayers, andtrainboth
201
theLoRAadaptersandtheTrajGazeMergeencoderjointlywithapurecross-entropylossonthe
202
merged-tokenstudentlogits,L =L (pstud,y). GradientsflowthroughtheVLMLoRA,through
203 S2 CE
thedifferentiablescore-weightedmergeoperation,andbackintotheTrajGazeMergeencoder’spatch
204
scores,enablingend-to-endoptimizationofthefullselectionpipelinewithoutanexternalteacher.
205
4 ExperimentalDetails
206
4.1 TrainingandEvaluationProtocol
207
WeusetheStreamGazebenchmark[45],whichisbuiltfromthreeegocentricvideosources: EgoEx-
208
oLearn[46],HoloAssist[5],andEGTEA[18]. Followingthissourcecomposition,wetrainonthe
209
EgoExoLearn and HoloAssist portions and evaluate on the disjoint EGTEA portion. We use all
210
multiple-choicequestionsfromtheselectedtrainingandevaluationportions.
211
Formethodsthatrequiretraining, includingTrajGazeMergeandtrainablebaselineselectors, the
212
selectoristrainedontheEgoExoLearnandHoloAssisttrainingportion. Forzero-shotevaluations,we
213
applyeachselectortotheLoRA-finetunedVLMandretaintheprescribedfractionofvisualtokens.
214
Forfine-tuningevaluations,wejointlytraintheselectorandtheVLMLoRAadaptersunderthesame
215
datasplitandtokenbudget. Detailsregardingbaselinemodelsandtheirimplementationsaredetailed
216
intheAppendix.
217
4.2 Dataset
218
FollowingStreamGaze’sevaluationprotocol[45],weevaluateoneightegocentricVQAtasks. Gaze
219
Sequence Matching (GSM) asks the model to identify which of four candidate gaze transition
220
sequencesbestmatchestheuser’sobservedfixationpattern,requiringfine-grainedtemporalunder-
221
standingofattentiondynamics. Non-FixatedIdentification(NFI)presentsasetofobjectsandasks
222
whichonetheuserneverfixated, testingthemodel’sabilitytoreasonabouttheabsenceofgaze.
223
SceneRecall(SR)probesmemoryofbackgroundcontextbyaskingwhichobjectwasorwasnot
224
visibleduringaspecificgazeepisode. ObjectAttributeRecognition(OAR)concernsthematerial,
225
color,orstateoftheobjectcurrentlybeingfixatedon,groundingattributequeriesinreal-timegaze.
226
ObjectIdentification—Easy(OI-E)andHard(OI-H)bothaskwhichobjecttheuseriscurrently
227
gazingat,withthehardvariantusingvisuallysimilardistractorsthatrequirespatialprecision. Future
228
ActionPrediction(FAP)predictstheuser’snextactionbasedontherecentfixationsequence,linking
229
gazepatternstointentionalbehavior.
230
5 Results
231
5.1 QuantitativeComparison
232
Table1reportsperformanceacrosseightegocentricVQAtasksundertwovisualtokenbudgets(5%,
233
10%)andtwoevaluationprotocols. Werefertotheprotocolwhereaseparatelytrainedselectoris
234
appliedontopofaLoRA-finetunedbackboneasthefrozen-selectorsetting(the“Zero-shot”rows
235
inTable1),andtheprotocolwheretheselectorandtheVLMLoRAarejointlytrainedasthejoint
236
setting(the“LoRAfinetune”rows).
237
Joint setting at 5% budget. TrajGazeMerge achieves the best overall average, outperforming
238
PruneVidby+1.71pp,therule-basedgaze/handprunerby+3.23pp,FastVIDby+3.80pp,Auto-
239
Gazeby+7.80pp,andVisionZipby+9.88pp. ThemarginoverPruneVidismodestinaggregate,
240
6

Table 1: Results across visual token budgets (5%, 10%) under zero-shot and LoRA finetuning
protocols. AllmethodsutilizeQwen2.5-VL-7B[55]asthebackbone. Inthezero-shotprotocol,we
evaluateboththefrozenbasemodelandtheLoRA-finetunedmodelasbackbonesfortraining-free
reduction. Bestpercolumnwithineachsub-blockisbolded.
| Tokens Protocol | Method       | GSM NFI     | SR OAR      | OI(E) OI(H) | FAP Avg.    |
| --------------- | ------------ | ----------- | ----------- | ----------- | ----------- |
|                 |              |             | 80.21       | 59.38       |             |
|                 | PruneVid[37] | 37.50 54.41 | 35.14       | 55.45       | 26.60 49.81 |
|                 | AutoGaze[16] | 42.19 33.82 | 32.43 73.96 | 42.57 48.44 | 20.21 41.95 |
Zero-shot VisionZip[40] 50.94 45.59 48.65 47.92 52.48 46.88 32.98 46.49
|     | FastVID[34]         | 35.94 44.12 | 32.43 78.12 | 51.49 57.81 | 25.53 46.49 |
| --- | ------------------- | ----------- | ----------- | ----------- | ----------- |
|     | TrajGazeMerge(Ours) | 42.19 60.29 | 45.95 80.21 | 63.37 59.38 | 26.60 54.00 |
5%
|     | Rule-basedgaze/handpruner | 62.50 58.82 | 48.65 88.54 | 62.38 62.50 | 50.00 61.91 |
| --- | ------------------------- | ----------- | ----------- | ----------- | ----------- |
|     | PruneVid[37]              | 75.00 52.94 | 59.46 87.50 | 59.41 67.19 | 50.00 64.50 |
|     | AutoGaze[16]              | 59.38 48.53 | 45.95 88.54 | 49.50 62.50 | 47.87 57.47 |
LoRAfinetune VisionZip[40] 59.38 51.47 48.65 69.79 66.34 60.94 35.11 55.95
|     | FastVID[34]         | 62.50 60.29 | 54.05 86.46 | 58.42 65.62 | 47.87 62.17 |
| --- | ------------------- | ----------- | ----------- | ----------- | ----------- |
|     | TrajGazeMerge(Ours) | 68.75 66.18 | 51.35 88.54 | 62.38 70.31 | 52.13 65.66 |
|     | PruneVid[37]        | 43.75 58.82 | 43.24 81.25 | 56.44 60.94 | 27.66 53.16 |
|     | AutoGaze[16]        | 31.25 33.82 | 21.62 72.92 | 42.57 57.81 | 29.79 41.40 |
Zero-shot VisionZip[40] 64.06 42.65 54.05 44.79 57.43 56.25 32.98 50.32
|     | FastVID[34]         | 62.50 61.76 | 37.84 41.67 | 71.29 68.75 | 46.81 55.80 |
| --- | ------------------- | ----------- | ----------- | ----------- | ----------- |
|     | TrajGazeMerge(Ours) | 45.31 64.71 | 43.24 78.13 | 64.36 65.63 | 25.53 55.27 |
10% Fullvisualtoken 60.90 66.20 64.90 85.40 68.30 70.30 35.10 64.44
|     | Rule-basedgaze/handpruner | 62.50 55.88 | 48.65 90.62 | 64.36 64.06 | 42.55 61.23 |
| --- | ------------------------- | ----------- | ----------- | ----------- | ----------- |
|     | PruneVid[37]              | 59.38 57.35 | 48.65 89.58 | 67.33 70.31 | 50.00 63.23 |
LoRAfinetune AutoGaze[16] 54.69 50.00 51.35 87.50 46.53 42.55 42.55 53.60
|     | VisionZip[40]       | 59.38 52.94 | 51.35 65.62 | 65.35 65.62 | 35.11 56.48 |
| --- | ------------------- | ----------- | ----------- | ----------- | ----------- |
|     | FastVID[34]         | 64.06 63.24 | 54.05 84.38 | 63.37 67.19 | 44.68 63.00 |
|     | TrajGazeMerge(Ours) | 68.75 64.71 | 56.76 90.62 | 68.32 76.56 | 47.87 67.66 |
241 butTrajGazeMergeleadsmoresubstantiallyontasksthatdemandfine-grainedspatialreasoning:
Non-FixatedIdentification(+13.24ppoverPruneVid),ObjectIdentificationHard(+3.12pp),and
242
Future Action Prediction (+2.13 pp). The rule-based pruner is competitive on Object Attribute
243
Recognition,matchingTrajGazeMergeandAutoGazeonOAR,suggestingthatsimpleheuristics
244
245 sufficeforattributequeriesbutfallshortontasksrequiringtemporalorcompositionalreasoning.
Frozen-selectorvs.jointat5%budget. Thefrozen-selectorvariantofTrajGazeMergetrailsits
246
247 jointcounterpartby−11.59pp(54.95vs.66.54),andacomparablegapappearsforotherselectors
248 pairedwiththeLoRA-finetunedbackbone(e.g.,PruneVid: −13.50pp). Combiningtask-specific
LoRAfinetuningwithanindependentlytrainedselectorintroducesadistributionalmismatchbetween
249
thebackboneandtheselectorthatdegradesselectionquality. Jointtrainingisthereforeessentialfor
250
realizingthebenefitsoftrajectory-conditionedselection.
251
Frozen-selectorat10%budget. Withalargertokenbudget,thefrozen-selectorvariantofTra-
252
jGazeMergenarrowlyedgesFastVID(+0.01pp)andfrozenPruneVid(+2.10pp). Themodestgain
253
254 overthe5%frozen-selectorsettingsuggeststhat,withoutjointtraining,theselectoritselfbecomes
255 thebottleneck: doublingthetokenbudgetdoesnothelpiftheaddedtokenscarrylittletask-relevant
information.
256
257 Joint setting at 10% budget. Joint training of the selector and VLM LoRA yields the largest
absolutegainsforTrajGazeMerge. TrajGazeMergesurpassesPruneVidby+3.42pp,FastVIDby
258
+4.75pp,therule-basedprunerby+5.89pp,and—critically—thefull-tokenbaselineby+4.38pp.
259
Exceeding the full-token baseline at only 10% of visual tokens confirms that TrajGazeMerge’s
260
selectionisnotmerelylosslesscompressionbutactivelydenoisesthevisualcontextbysuppressing
261
262 uninformativebackgroundregions. VisionZipandAutoGazelagconsiderablyunderthisprotocol
263 (−11.79 pp and −13.83 pp behind TrajGazeMerge, respectively), suggesting that their selection
criteria—visualattentionstatisticsandgazesaliency—arelesswell-calibratedtotask-relevanttoken
264
importance.
265
Per-taskobservations. ObjectTransitionPrediction(OTP)containsonlytwotestitems,soper-
266
methodscoresfallonacoarse{0%,50%,100%}gridandwedrawnoconclusionsfromthiscolumn.
267
On Object Attribute Recognition (OAR), most LoRA-fine-tuned methods cluster in the 84–91%
268
7

Q. Which object is the user currently gazing at?
(a) Sausage (b) Stove (c) Plate (d) Glass Selected patch
(b) Stove
PruneVid off-target off-target
scattered (c) Plate
VisionZip
Ignore gaze
(a) Sausage
Ours
Correct
Figure2: QualitativecomparisonoftokenselectionmethodsonavideoQAexample. PruneVid
attendstooff-targetregionsandVisionZipyieldsscatteredpatchesignoringthegaze,whileTrajGaze-
Mergeconsistentlyfocusesonthegazedobject(sausage)acrossframesandproducesthecorrect
answer.
range,suggestingattributequeriesareanswerablefromsparsevisualevidence;themainexception
269
isVisionZip(65.62%),whosecontent-basedselectionappearstoscattertokenstoowidely. Future
270
ActionPrediction(FAP)isthehardesttask: allmethodssitwellabovefour-waychance(25%)but
271
wellbelowceiling,rangingfrom35.11%(VisionZip)to50.00%(PruneVid).
272
5.2 QualitativeComparison
273
Figure2illustratesthequalitativebasisforthequantitativegainsreportedinTable1. Onagazed-
274
objectidentificationquery(“Whichobjectistheusercurrentlygazingat?” withcandidatessausage,
275
stove, plate, and glass), PruneVid scatters its token budget across small off-target clusters near
276
theperipheryofthescene;itstext–imagesimilarityscoringlatchesontogenericstove-likevisual
277
conceptsandthemodelanswers“stove.” VisionZipallocatestokensalonghigh-contrastdiagonal
278
contoursthattraversetheentireframe,producingaspatiallydiffuseselectionthatneverconcentrates
279
ontheactor’spointoffixation,andthemodelanswers“plate.” TrajGazeMerge,bycontrast,produces
280
acompact,contiguousclusteroftokenstightlylocalizedonthesausageineveryframeandtracks
281
theobjectastheegocentricviewpointshifts,yieldingthecorrect“sausage”answer. Twoproperties
282
ofourdesignaccountforthisbehavior. First,conditioningthescoringnetworkonhandandgaze
283
trajectoriesanchorsselectiontotheactor’sbehavioralfocusratherthantobottom-upsaliencyor
284
text-image similarity—both of which are only weakly correlated with the question’s referent in
285
egocentricvideo,wherethesalientortext-matchedregions(stoveedges,platerims)frequentlydiffer
286
fromtheactuallyattendedobject. Second,thespatio-temporalmergingsteppropagatesthisanchor
287
coherentlyacrossframes,sothatevenwhenthegazedobjectdriftsinpixelspaceduetoheadmotion,
288
theselectedregionremainsstableandcontiguousratherthanfragmentingintoisolatedpatches. The
289
visualevidencethusmirrorsthequantitativetrends: byaligningthetokenbudgetwiththeegocentric
290
viewer’sactualfocusratherthanwithsurrogatevisualsignals,TrajGazeMergesuppliestheVLM
291
withpreciselythepatchesneededtoanswerthequestion.
292
5.3 Ablationstudies
293
Table2: AblationonthecontributionofhandandgazemodalitiesforTrajGazeMerge. (Note: Object
TransitionPrediction(OTP)isexcludedasallablatedvariantsscored0.00%.)
Modality GSM NFI SR OAR OI(E) OI(H) FAP Avg.
OnlyHand 71.88 58.82 62.16 90.62 62.38 65.62 48.94 66.16
OnlyGaze 68.75 58.82 56.76 89.58 56.44 68.75 50.00 64.64
Hand+Gaze 68.75 64.71 56.76 90.62 68.32 76.56 47.87 68.44
Pretraining objectives. Table 3 ablates TrajGazeMerge’s pretraining losses. Score regression
294
aloneimprovesuponno-pretrainingby+0.76pp, notablyaidingtemporallystructuredtaskslike
295
8

Table3: AblationonpretrainingobjectivesforTrajGazeMerge.
Pretraining GSM NFI OTP SR OAR OI(E) OI(H) FAP Avg.
Nopretrain 65.62 63.24 0.00 51.35 86.46 61.39 68.75 52.13 65.02
Onlyscoreloss 76.56 61.76 50.00 56.76 89.58 59.41 67.19 46.81 65.78
Allloss 68.75 64.71 50.00 56.76 90.62 68.32 76.56 47.87 68.44
Table 4: Ablation on spatial vs. temporal pruning for TrajGazeMerge. (Note: Object Transition
Prediction(OTP)isexcludedasallvariantsscored0.00%.)
Pruning GSM NFI SR OAR OI(E) OI(H) FAP Avg.
Nospatial 67.19 60.29 45.95 88.54 61.39 65.63 51.06 64.64
Notemporal 51.56 54.41 56.76 82.29 66.34 62.50 48.94 61.43
Spatio-temporal 75.00 61.76 56.76 87.50 64.36 71.88 52.13 67.49
GSM(+10.94pp)andOTP(+50.00pp). Fullpretraining(score+auxiliarylosses)achievesthebest
296
overallperformance(+3.42ppoverno-pretraining),stronglybenefitingfine-grainedspatialtaskslike
297
OI-H(+9.37pp)andOI-E(+8.91pp). However,fullpretrainingslightlyregressesGSMrelativeto
298
score-only(−7.81pp),highlightingadelicatetrade-offbetweencapturingcoarsetemporalrhythms
299
andfine-grainedspatialprecision.
300
Gazevs.handmodality. Table2isolatesegocentriccues. Hand-onlyscoringoutperformsgaze-
301
only(+1.52ppoverall),dominatingmanipulation-heavytaskslikeGSM(+3.13pp)andSR(+5.40pp).
302
Conversely,gaze-onlyexcelsinFAP(+1.06pp)andOI-H(+3.13pp),offeringpredictivecuesfor
303
intention and attended objects. Fusing both modalities yields the best average (+2.28 pp over
304
hand-only),withmassivegainsinhard-objectidentification(OI-H,+10.94pp). Thisconfirmstheir
305
complementarity: handsphysicallygroundcurrentactions,whilegazerevealsanticipatedbroader
306
context.
307
Spatial vs. temporal pruning. Table 4 evaluates selection components. Disabling temporal
308
pruning(“Notemporal”)degradesperformanceseverely(−6.06ppoverall;−23.44pponGSM),
309
demonstratingthatretainingirrelevantframesfloodsthemodelwithuninformativecontent. Disabling
310
spatialpruningislessdamagingbutstillsub-optimal(−2.85ppoverall).Fullspatio-temporalpruning
311
peakson6of7tasks, provingbothaxesareessential: temporalselectionisolatessalientframes,
312
whilespatialpruningremovesintra-framebackgroundclutter.
313
6 Conclusion
314
We presented TrajGazeMerge, a behavior-conditioned token compression framework that turns
315
the gaze and bimanual hand trajectories naturally accompanying the egocentric video into the
316
patch-level importance signals for video MLLMs. By jointly modeling gaze, hands, and their
317
interactionkinematicsthroughatrajectory-conditionedencoder,andbyusingtheresultingscores
318
to drive differentiable score-weighted bipartite token merging, our method retains behaviorally
319
salient evidence under aggressive compression budgets where generic visual statistics or query-
320
frame relevance tend to discard task-critical regions. On StreamGaze, trained on EgoExoLearn
321
and HoloAssist, and evaluated on the disjoint EGTEA portion, TrajGazeMerge reaches 68.44%
322
overallaccuracyatonly10%visual-tokenretention,surpassingattention-basedpruning,content-
323
basedmerging,frame-selectionmethods,andnotablyexceedingthefull-tokenLoRAbaselineby
324
+4.38 pp. Ablations confirm that gaze and hand provide complementary evidence (fusing both
325
yieldsthelargestgainsonhardobjectidentification),thatthefour-objectivetrajectorypretraining
326
contributessubstantiallytofine-grainedspatialdiscrimination,andthatjointspatio-temporalselection
327
isessential–removingeitheraxissharplydegradesperformance. Together,theseresultsindicate
328
thattheegocentrictokenbottleneckisbestaddressednotonlybygenericcompressionbutalsoby
329
conditioningcompressiononthebehavioralsignalsalreadypresentinthestream.
330
9

References
331
[1] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari,
332
EvangelosKazakos,DavideMoltisanti,JonathanMunro,TobyPerrett,WillPrice,etal. Scaling
333
egocentricvision: Theepic-kitchensdataset. InProceedingsoftheEuropeanConferenceon
334
ComputerVision(ECCV),pages720–736,2018.
335
[2] DimaDamen,HazelDoughty,GiovanniMariaFarinella,AntoninoFurnari,JianMa,Evangelos
336
Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray.
337
Rescalingegocentricvision: Collection,pipelineandchallengesforEPIC-KITCHENS-100.
338
InternationalJournalofComputerVision(IJCV),130:33–55,2022.
339
[3] KristenGrauman,AndrewWestbury,EugeneByrne,ZacharyChavis,AntoninoFurnari,Rohit
340
Girdhar,JacksonHamburger,HaoJiang,MiaoLiu,XingyuLiu,etal. Ego4d: Aroundtheworld
341
in3,000hoursofegocentricvideo. InProceedingsoftheIEEE/CVFConferenceonComputer
342
VisionandPatternRecognition,pages18995–19012,2022.
343
[4] KristenGrauman,AndrewWestbury,LorenzoTorresani,KrisKitani,JitendraMalik,Triantafyl-
344
losAfouras,KumarAshutosh,VijayBaiyya,SiddhantBansal,BikramBoote,etal. Ego-exo4d:
345
Understandingskilledhumanactivityfromfirst-andthird-personperspectives. InProceedings
346
oftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages19383–19400,
347
2024.
348
[5] XinWang,TaeinKwon,MahdiRad,BowenPan,IshaniChakraborty,SeanAndrist,DanBohus,
349
AshleyFeniello,BugraTekin,FelipeVieiraFrujeri,NeelJoshi,andMarcPollefeys. Holoassist:
350
anegocentrichumaninteractiondatasetforinteractiveaiassistantsintherealworld,2023. URL
351
https://arxiv.org/abs/2309.17024.
352
[6] Jean-BaptisteAlayrac, JeffDonahue, PaulineLuc, AntoineMiech, IainBarr, YanaHasson,
353
KarelLenc,ArthurMensch,KatieMillican,MalcolmReynolds,RomanRing,ElizaRuther-
354
ford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Ja-
355
cob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh,
356
Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Si-
357
monyan. Flamingo: A visual language model for few-shot learning, 2022. URL https:
358
//arxiv.org/abs/2204.14198.
359
[7] HaotianLiu,ChunyuanLi,QingyangWu,andYongJaeLee. Visualinstructiontuning,2023.
360
URLhttps://arxiv.org/abs/2304.08485.
361
[8] YanweiLi,ChengyaoWang,andJiayaJia. LLaMA-VID:Animageisworth2tokensinlarge
362
languagemodels. arXivpreprintarXiv:2311.17043,2023.
363
[9] PengJin,RyuichiTakanobu,WancaiZhang,XiaochunCao,andLiYuan. Chat-UniVi: Unified
364
visualrepresentationempowerslargelanguagemodelswithimageandvideounderstanding. In
365
IEEE/CVFConferenceonComputerVisionandPatternRecognition(CVPR),2024.
366
[10] PengWang,ShuaiBai,SinanTan,ShijieWang,ZhihaoFan,JinzeBai,KeqinChen,Xuejing
367
Liu,JialinWang,WenbinGe,YangFan,KaiDang,MengfeiDu,XuanchengRen,RuiMen,
368
DayihengLiu,ChangZhou,JingrenZhou,andJunyangLin. Qwen2-VL:Enhancingvision-
369
languagemodel’sperceptionoftheworldatanyresolution. arXivpreprintarXiv:2409.12191,
370
2024.
371
[11] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava:
372
Learning united visual representation by alignment before projection, 2024. URL https:
373
//arxiv.org/abs/2311.10122.
374
[12] BaoxiongJia,TingLei,Song-ChunZhu,andSiyuanHuang. Egotaskqa: Understandinghuman
375
tasksinegocentricvideos. AdvancesinNeuralInformationProcessingSystems,35:3343–3360,
376
2022.
377
[13] KarttikeyaMangalam,RaiymbekAkshulakov,andJitendraMalik. EgoSchema: Adiagnostic
378
benchmarkforverylong-formvideolanguageunderstanding.InAdvancesinNeuralInformation
379
ProcessingSystems(NeurIPS),2024.
380
10

[14] Hanrong Ye, Haotian Zhang, Erik Daxberger, Lin Chen, Zongyu Lin, Yanghao Li, Bowen
381
Zhang,HaoxuanYou,DanXu,ZheGan,ShuZhang,ChunyuanLi,De-AnHuang,Zhiding
382
Yu, Sifei Liu, Guilin Liu, Pavlo Molchanov, Jan Kautz, Serge Belongie, and Ming-Hsuan
383
Yang. MM-Ego: TowardsbuildingegocentricmultimodalLLMsforvideoQA,2024. URL
384
https://arxiv.org/abs/2410.07177.
385
[15] Taiying Peng, Jiacheng Hua, Miao Liu, and Feng Lu. In the eye of mllm: Benchmarking
386
egocentric video intent understanding with gaze-guided prompting, 2025. URL https://
387
arxiv.org/abs/2509.07447.
388
[16] BaifengShi,StephanieFu,LongLian,HanrongYe,DavidEigen,AaronReite,BoyiLi,Jan
389
Kautz,SongHan,DavidM.Chan,PavloMolchanov,TrevorDarrell,andHongxuYin. Attend
390
beforeattention: Efficientandscalablevideounderstandingviaautoregressivegazing,2026.
391
URLhttps://arxiv.org/abs/2603.12254.
392
[17] DandanShan,JiaqiGeng,MichelleShu,andDavidFFouhey. Understandinghumanhandsin
393
contactatinternetscale. InProceedingsoftheIEEE/CVFConferenceonComputerVisionand
394
PatternRecognition,pages9869–9878,2020.
395
[18] YinLi,MiaoLiu,andJamesM.Rehg. Intheeyeofbeholder:Jointlearningofgazeandactions
396
infirstpersonvideo. InEuropeanConferenceonComputerVision(ECCV),2018.
397
[19] MengmiZhang,KengTeckMa,JooHweeLim,QiZhao,andJiashiFeng. Deepfuturegaze:
398
Gazeanticipationonegocentricvideosusingadversarialnetworks. InProceedingsoftheIEEE
399
ConferenceonComputerVisionandPatternRecognition,pages4372–4381,2017.
400
[20] YifeiHuang,MinjieCai,ZhenqiangLi,andYoichiSato. Predictinggazeinegocentricvideo
401
bylearningtask-dependentattentiontransition. InEuropeanConferenceonComputerVision
402
(ECCV),2018.
403
[21] Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer, and Kristen Grauman. Ego-topo:
404
Environmentaffordancesfromegocentricvideo. InIEEE/CVFConferenceonComputerVision
405
andPatternRecognition(CVPR),2020.
406
[22] ShaoweiLiu,SubarnaTripathi,SomdebMajumdar,andXiaolongWang. Jointhandmotion
407
andinteractionhotspotspredictionfromegocentricvideos. InProceedingsoftheIEEE/CVF
408
ConferenceonComputerVisionandPatternRecognition,pages3282–3292,2022.
409
[23] MichaelFLandandMaryHayhoe. Inwhatwaysdoeyemovementscontributetoeveryday
410
activities? VisionResearch,41(25-26):3559–3565,2001.
411
[24] Mary Hayhoe and Dana Ballard. Eye movements in natural behavior. Trends in Cognitive
412
Sciences,9(4):188–194,2005.
413
[25] LaurentItti,ChristofKoch,andErnstNiebur. Amodelofsaliency-basedvisualattentionfor
414
rapidsceneanalysis. IEEETransactionsonPatternAnalysisandMachineIntelligence,20(11):
415
1254–1259,1998.
416
[26] LaurentIttiandChristofKoch. Computationalmodellingofvisualattention. NatureReviews
417
Neuroscience,2(3):194–203,2001.
418
[27] YongmingRao,WenliangZhao,BenlinLiu,JiwenLu,JieZhou,andCho-JuiHsieh.Dynamicvit:
419
Efficientvisiontransformerswithdynamictokensparsification.AdvancesinNeuralInformation
420
ProcessingSystems,34:13937–13949,2021.
421
[28] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not
422
allpatchesarewhatyouneed: Expeditingvisiontransformersviatokenreorganizations. In
423
InternationalConferenceonLearningRepresentations, 2022. URLhttps://arxiv.org/
424
abs/2202.07800.
425
[29] DanielBolya,Cheng-YangFu,XiaoliangDai,PeizhaoZhang,ChristophFeichtenhofer,and
426
JudyHoffman. Tokenmerging: Yourvitbutfaster. InInternationalConferenceonLearning
427
Representations,2023. URLhttps://arxiv.org/abs/2210.09461.
428
11

[30] HongxuYin,ArashVahdat,JoseM.Alvarez,ArunMallya,JanKautz,andPavloMolchanov.
429
A-ViT:Adaptivetokensforefficientvisiontransformer. InIEEE/CVFConferenceonComputer
430
VisionandPatternRecognition(CVPR),2022.
431
[31] JindongJiang,XiuyuLi,ZhijianLiu,MuyangLi,GuoChen,ZhiqiLi,De-AnHuang,Guilin
432
Liu,ZhidingYu,KurtKeutzer,etal. STORM:Token-efficientlongvideounderstandingfor
433
multimodalLLMs. InProceedingsoftheIEEE/CVFInternationalConferenceonComputer
434
Vision,pages5830–5841,2025.
435
[32] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao,
436
KunchangLi,YinanHe,ChentingWang,etal. VideoChat-Flash: Hierarchicalcompressionfor
437
long-contextvideomodeling,2024. URLhttps://arxiv.org/abs/2501.00574.
438
[33] XiTang,JihaoQiu,LingxiXie,YunjieTian,JianbinJiao,andQixiangYe. Adaptivekeyframe
439
sampling for long video understanding. In Proceedings of the IEEE/CVF Conference on
440
ComputerVisionandPatternRecognition,pages29118–29127,2025.
441
[34] Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, and
442
Guiguang Ding. FastVID: Dynamic density pruning for fast video large language models,
443
2025. URLhttps://arxiv.org/abs/2503.11187.
444
[35] XiaoqianShen,YunyangXiong,ChangshengZhao,LemengWu,JunChen,ChenchenZhu,
445
Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu,
446
HyunwooJKim,BilgeSoran,RaghuramanKrishnamoorthi,MohamedElhoseiny,andVikas
447
Chandra.Longvu:Spatiotemporaladaptivecompressionforlongvideo-languageunderstanding.
448
InInternationalConferenceonMachineLearning,2025. URLhttps://arxiv.org/abs/
449
2410.17434.
450
[36] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao
451
Chang. Animageisworth1/2tokensafterlayer2: Plug-and-playinferenceaccelerationfor
452
largevision-languagemodels. InProceedingsoftheEuropeanConferenceonComputerVision
453
(ECCV),2024. URLhttps://arxiv.org/abs/2403.06764.
454
[37] XiaohuHuang,HaoZhou,andKaiHan. Prunevid: Visualtokenpruningforefficientvideo
455
largelanguagemodels,2024. URLhttps://arxiv.org/abs/2412.16117.
456
[38] ShaojieZhang,JiahuiYang,JianqinYin,ZhenboLuo,andJianLuan. Q-frame: Query-aware
457
frameselectionandmulti-resolutionadaptationforvideo-llms. InProceedingsoftheIEEE/CVF
458
InternationalConferenceonComputerVision(ICCV),pages22056–22065,October2025.
459
[39] Hui Sun, Shiyin Lu, Huanyu Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang,
460
andMingLi. MDP3: Atraining-freeapproachforlist-wiseframeselectioninvideo-llms. In
461
ProceedingsoftheIEEE/CVFInternationalConferenceonComputerVision(ICCV),pages
462
24090–24101,October2025.
463
[40] SenqiaoYang,YukangChen,ZhuotaoTian,ChengyaoWang,JingyaoLi,BeiYu,andJiaya
464
Jia. VisionZip: Longerisbetterbutnotnecessaryinvisionlanguagemodels. InProceedingsof
465
theIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages19792–19802,
466
2025.
467
[41] YuzhangShang,MuCai,BingxinXu,YongJaeLee,andYanYan. Llava-prumerge: Adaptive
468
token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF
469
InternationalConferenceonComputerVision(ICCV),pages22857–22867,2025.
470
[42] YuanZhang,Chun-KaiFan,JunpengMa,WenzhaoZheng,TaoHuang,KuanCheng,Denis
471
Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token
472
sparsificationforefficientvision-languagemodelinference. InInternationalConferenceon
473
MachineLearning,2025. URLhttps://arxiv.org/abs/2410.04417.
474
[43] LongXing,QidongHuang,XiaoyiDong,JiajieLu,PanZhang,YuhangZang,YuhangCao,
475
ConghuiHe, JiaqiWang, FengWu, andDahuaLin. Pyramiddrop: Acceleratingyourlarge
476
vision-languagemodelsviapyramidvisualredundancyreduction,2025.URLhttps://arxiv.
477
org/abs/2410.17247.
478
12

[44] Anupam Pani and Yanchao Yang. Gaze-vlm: Bridging gaze and vlms through attention
479
regularizationforegocentricunderstanding. arXivpreprintarXiv:2510.21356,2025.
480
[45] DaeunLee,SubhojyotiMukherjee,BranislavKveton,RyanARossi,VietDacLai,Seunghyun
481
Yoon,TrungBui,FranckDernoncourt,andMohitBansal. Streamgaze: Gaze-guidedtemporal
482
reasoningandproactiveunderstandinginstreamingvideos. arXivpreprintarXiv:2512.01707,
483
2025.
484
[46] YifeiHuang,GuoChen,JilanXu,MingfangZhang,LijinYang,BaoqiPei,HongjieZhang,
485
LuDong,YaliWang,LiminWang,andYuQiao. Egoexolearn: Adatasetforbridgingasyn-
486
chronousego-andexo-centricviewofproceduralactivitiesinrealworld. InProceedingsof
487
theIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages22072–22086,
488
2024.
489
[47] YinLi,MiaoLiu,andJamesMRehg.Intheeyeofthebeholder:Gazeandactionsinfirstperson
490
video. IEEETransactionsonPatternAnalysisandMachineIntelligence,45(6):6731–6747,
491
2021. doi: 10.1109/TPAMI.2021.3051319.
492
[48] KevinQinghongLin,AlexWang,MattiaSoldan,MichaelWray,RuiYan,EricZhongcunXu,
493
DifeiGao,Ron-JuniorTu,KunchangZhao,LingpengKong,ChenGao,HaoJiang,MikeZheng
494
Shou,GedasBertasius,LorenzoTorresani,WanliGeng,WeiLiu,andMengmengLiu. Ego-
495
centricvideo-languagepretraining. InAdvancesinNeuralInformationProcessingSystems
496
(NeurIPS),2022.
497
[49] ShramanPramanick,YaleSong,SayanNag,MikeZhengShou,MubarakShah,andLeonid
498
Ferreira. EgoVLPv2: Egocentricvideo-languagepre-trainingwithfusioninthebackbone. In
499
IEEE/CVFInternationalConferenceonComputerVision(ICCV),2023.
500
[50] ShengZhou,JunbinXiao,QingyunLi,YicongLi,XunYang,DanGuo,MengWang,Tat-Seng
501
Chua, and Angela Yao. Egotextvqa: Towards egocentric scene-text aware video question
502
answering. InProceedingsoftheComputerVisionandPatternRecognitionConference,pages
503
3363–3373,2025.
504
[51] SimoneAlbertoPeirone,FrancescaPistilli,andGiuseppeAverta. Hiero: Understandingthe
505
hierarchyofhumanbehaviorenhancesreasoningonegocentricvideos. InProceedingsofthe
506
IEEE/CVFInternationalConferenceonComputerVision,pages19862–19871,2025.
507
[52] ShulinTian,RuiqiWang,HongmingGuo,PenghaoWu,YuhaoDong,XiuyingWang,Jingkang
508
Yang,HaoZhang,HongyuanZhu,andZiweiLiu. Ego-r1: Chain-of-tool-thoughtforultra-long
509
egocentricvideoreasoning,2025. URLhttps://arxiv.org/abs/2506.13654.
510
[53] EthanPerez,FlorianStrub,HarmDeVries,VincentDumoulin,andAaronCourville. FiLM:
511
Visualreasoningwithageneralconditioninglayer.InAAAIConferenceonArtificialIntelligence,
512
2018.
513
[54] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil
514
Khalidov,PierreFernandez,DanielHaziza,FranciscoMassa,AlaaeldinEl-Nouby,Mahmoud
515
Assran,NicolasBallas,WojciechGaluba,RussellHowes,Po-YaoHuang,Shang-WenLi,Ishan
516
Misra,MichaelRabbat,VasuSharma,GabrielSynnaeve,HuXu,HervéJegou,JulienMairal,
517
Patrick Labatut, Armand Joulin, and Piotr Bourdoukan. DINOv2: Learning robust visual
518
featureswithoutsupervision. TransactionsonMachineLearningResearch(TMLR),2024.
519
[55] ShuaiBai,KeqinChen,XuejingLiu,JialinWang,WenbinGe,SiboSong,KaiDang,PengWang,
520
ShijieWang,JunTang,HumenZhong,YuanzhiZhu,MingkunYang,ZhaohaiLi,Jianqiang
521
Wan,PengfeiWang,WeiDing,ZherenFu,YihengXu,JiaboYe,XiZhang,TianbaoXie,Zesen
522
Cheng,HangZhang,ZhiboYang,HaiyangXu,andJunyangLin. Qwen2.5-VLtechnicalreport.
523
arXivpreprintarXiv:2502.13923,2025.
524
[56] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
525
LuWang,andWeizhuChen. Lora: Low-rankadaptationoflargelanguagemodels,2021. URL
526
https://arxiv.org/abs/2106.09685.
527
13

[57] IlyaLoshchilovandFrankHutter. Decoupledweightdecayregularization. InInternational
528
ConferenceonLearningRepresentations,2019.
529
[58] LucaBarsellotti,LorenzoBianchi,NicolaMessina,FabioCarrara,MarcellaCornia,Lorenzo
530
Baraldi,FabrizioFalchi,andRitaCucchiara. TalkingtoDINO:Bridgingself-supervisedvision
531
backboneswithlanguageforopen-vocabularysegmentation. InProceedingsoftheIEEE/CVF
532
InternationalConferenceonComputerVision(ICCV),pages22025–22035,2025.
533
A ImplementationDetails
534
OursystemistrainedintwostagesonasingleNVIDIAH200GPU.Stage1pretrainsthetrajectory
535
encoder on StreamGaze [45] with trajectory-prediction and patch-importance objectives (§A.2).
536
Stage2jointlyfine-tunesLoRAadaptersontheVLMandthetrajectoryencoderthroughthetoken-
537
mergingmodule(§A.3). Wefirstsummarizethetrajectoryencoderarchitectureandparameterbudget
538
(§A.1);thefulldesignisdescribedin§??.
539
A.1 TrajectoryEncoder: ArchitectureandParameterBudget
540
Thetrajectoryencoderhas36.11Mtotalparameters,ofwhich14.05Maretrainable. Thefrozen
541
DINOv2-S/14 visual backbone [54] (∼22.1M parameters, patch size 14, embedding dimension
542
384)accountsforthedifference. Modality-specificwidthsared =128,d =256,d =256,and
543 traj enc vis
d =128. Thecomponentbreakdownisasfollows.
544 query
• Queryencoder(1.46M):a2-layer,4-headtransformerovertokenizedquestiontextwith
545
sinusoidalpositionalencoding,pooledtoasingle128-dqueryembedding. DuringStage1,
546
thequeryencoderisnotinvoked;thedownstreamFiLMblockreceivesafixedzerovector
547
asthequeryembeddingbecausenoquestion-answerlabelsareavailable. InStage2,the
548
queryencoderprocessestheactualquestiontextandistrainedjointlywiththerestofthe
549
trajectoryencoder.
550
• Visual patch encoder (22.16M total; 0.10M trainable): frozen DINOv2-S/14 over 16
551
keyframesresizedto196×196produces14×14=196patchtokensat384d,followedby
552
atrainablelinearprojectiontod =256andtemporallinearinterpolationthatalignsthe
553 vis
visualpatchtokenstotheT trajectoryframes.
554
• Spatiotemporaltrajectoryencoder(5.56M):aper-frametrajectorytokenizerproducing
555
fourtokens{zg,zL,zR,zϕ}forgaze,lefthand,righthand,andgaze-handinteraction;an
556
intra-frametransformer(L1: 2layers,4heads,d =128,FFN4d);aninter-frametemporal
557 traj
transformer (L2: 6 layers, 8 heads, d =256, FFN 4d, sinusoidal positional encoding)
558 enc
wrappedinatanh-gatedresidualx =x+tanh(g)(x −x)withasinglelearnable
559 main inter
scalarg;aFiLM[53]blockconditionedonthequeryembedding;andaper-framevisual-
560
trajectorycross-attentionfusionlayer.
561
562
• Patch-temporalmodulationbranch(0.31M):196learnedpatchqueriesQ
pat
∈R196×denc
563
cross-attendtotheinter-frametransformer’sper-frameoutputx
inter
∈RB×T×4×denc using
564
asingle-layer4-headattentionmodulewithdropout0.1, followedby LN+LINEAR+σ.
Thisbranchproducesaper-frame,per-patchmodulationmapM∈[0,1]B×T×196,which
565
ismultipliedelement-wiseintotheper-framepatchscoresemittedbythevisual-trajectory
566
fusion. Thebranchconsumesx independentlyofthemainresidualgate. Thegategis
567 inter
initializedto0andfrozenthroughoutStage1,sox =xandtheinter-frameoutputserves
568 main
onlyasthekey-valueinputofthismodulationbranch;inStage2,gisreleasedandupdated
569
jointlywiththerestofthetrajectoryencoder.
570
• Trajectorydecoder(3.20M):a3-layer,8-headcross-attentiondecoderoverfuture-frame
571
queries, projecting to a 6-d output corresponding to gaze, left-hand, and right-hand 2D
572
positionswithasigmoidoutput.
573
• Scoredecoder(3.31M):aparallel3-layer,8-headcross-attentiondecoderprojectingto
574
196-dpatchscoresperfutureframe;thisdecoderisusedasauxiliarysupervisioninStage1
575
only.
576
• TrajScoreHead(0.12M):anattentionpooloverthefourper-frametrajectorytokensfol-
577
lowedbyatwo-layerMLP(LayerNorm→Linear→GELU→Linear)withasigmoidhead
578
14

emittinga196-dpatch-importancemap. Thisoutputisgatedelement-wisebythepatch-
579
temporalmodulationmapM toproducethefinalper-framescoremapconsumedatinference
580
timeandbytheStage2tokenmerger.
581
A.2 Stage1: Trajectory-DrivenPatch-ImportancePretraining
582
Data. Stage 1 uses StreamGaze clips drawn from EgoExoLearn [46] and HoloAssist [5]. This
583
stageusesonlybehavioralsignals,namelygazeandhandtrajectories,togetherwithimageframes;
584
noquestion-answerlabelsareused. EachclipisuniformlysampledintoT=128framesandsplit
585
intoapast prefixoflengthT andafuturesuffixoflengthT −T ,wherethesplitratioT /T is
586 p p p
drawnuniformlyfrom[0.4,0.6]ateverytrainingstep. Thisrandomsplitforcesthemodeltosupport
587
differentobservedprefixlengthsatinferencetime. Gazeandhandpositionsareloadedinnormalized
588
imagecoordinateswithbinaryvisibilityflags. Missingmodalitiesarehandledbydedicatedlearnable
589
missingembeddingsinsidethetrajectorytokenizer.
590
WhyStreamGazefortrajectory-basedtokenreduction. WeevaluateonStreamGaze[45]be-
591
causeitprovideslongfirst-personstreamswithsynchronizedgazeandbimanualhandtracking,mak-
592
ingitwell-suitedforstudyingtrajectory-conditionedtokencompression. Sparse-framegaze-MLLM
593
benchmarkssuchasEgoGazeVQA[15]arehighlyrelevantforgaze-guidedintentunderstanding,but
594
theyprimarilyevaluategazecuesoverasmallsetofsampledframesratherthancontinuousgaze-hand
595
trajectories. StreamGazealsospansmultipleegocentricsources(EgoExoLearn, HoloAssist, and
596
EGTEA),enablingasource-disjointtrainandevaluationsplitforbehavior-conditionedcompression.
597
Per-frame supervision targets. For every frame t, we render a ground-truth 14×14 patch-
598
importancemapI ∈[0,1]196bysummingisotropicGaussianresponsesonthepatch-gridcenters.
599 t
ThetargetcontainsagazeGaussianwithbandwidthσ =16px,definedinthe224×224framecoor-
600 g
dinatesystem,andtheper-handmaximumoftwohandGaussianswithbandwidthσ =24px. Each
601 h
modalitycontributeszerowheneveritsvisibilityflagisoff. Theresultingmapisnormalizedper
602
frameandsplitintoIpast andIfuture .
603 1:Tp Tp+1:T
Objective. Thetrajectoryencoderistrainedend-to-endwithasumoffourMSEterms:
604
L = L + L + L + L . (9)
stage1 traj score-fut score-past score-traj
EachMSEtermisfirstaveragedoveritsvalidtemporalpositionsandoutputdimensions,andthefour
605
normalizedlossesarethensummedwithunitweights.
606
• L : future-trajectory regression in 2D image coordinates from the trajectory decoder,
607 traj
valid-length-masked over T −T frames. Gaze, left-hand, and right-hand positions are
608 p
stackedintoa6-dtarget.
609
• L : futureper-framepatch-scoreregressionfromthescoredecoderagainstIfuture.
610 score-fut
• L :regressionoftheencoder’srawvisualcross-attentionreadoutsagainstIpast,acting
611 score-past
asaspatial-groundingauxiliaryontheobservedpastframes.
612
• L : distillation of the trajectory-driven attention chain Atraj = (cid:80) decAttn ·
613 score-traj t,p t′ t′,t
encAttn intotheinference-timeTrajScoreHeadafterpatch-temporalmodulation. The
614 t,p
chaintargetisdetachedandmax-normalizedperframeto[0,1]. Thistermtrainsthepatch
615
scoresconsumedbyStage2.
616
Allfourtermsarecomputedwiththevariable-lengthvalidmaskinducedby(T ,T −T ). Thegateg
617 p p
isinitializedto0(tanh(g)=0)andfrozenduringStage1;onlythepatch-temporalmodulationbranch
618
readstheinter-frametransformer’soutput,whilethemaintrajectorypathbypassesit. Wefoundthat
619
thissingle-pathconfigurationavoidstheunstableinteractionbetweentheinter-frametransformerand
620
thevisual-trajectoryfusionobservedinearlyjointrandom-initializationablations.
621
Optimization. Stage1istrainedfor100epochswithAdamW[57](β =0.9,β =0.999,weight
622 1 2
decay10−4),peaklearningrate3×10−4,andacosine-annealingscheduledecayingto3×10−6,i.e.,
623
1% of the peak learning rate. The per-GPU batch size is 2, limited by the T=128 frame tensor,
624
yieldinganeffectiveglobalbatchsizeof2. Stage1runsonasingleNVIDIAH200GPUandtakes
625
approximately55minutesforthefullschedule. Framesareloadedat2242 andinternallyresized
626
15

to1962 ontheDINOv2pathsothatthe14×14patchgridalignswithI . Thecheckpointwiththe
627 t
lowestmeantraininglossisretainedastheStage2initialization.
628
A.3 Stage2: JointFine-TuningwithVLMandTokenMerging
629
Vision-languagebackboneandadapters. WeuseQwen2.5-VL-7B-Instruct[55]asthefrozen
630
vision-languagebackbone,loadedinbfloat16. Boththevisualencoderandthelanguagemodel
631
backbonearekeptfrozen;onlyLoRAadapters[56]aretrainedontheLLMside. LoRAisinjected
632
onlyintothefourself-attentionprojections{q ,k ,v ,o }ofeveryLLMtransformerlayer.
633 proj proj proj proj
Thevisual-encoderattentionlayersarenotadapted. Weuserankr=16,scalingα=32,dropout0.05,
634
and no bias term. This yields 10,092,544 trainable LoRA parameters out of 8,302,259,200 total
635
parameters (0.122%). The Stage 1 trajectory encoder is loaded into the same training graph and
636
fine-tunedend-to-endalongsidetheLoRAadapters.
637
VLMinputprompt. Foreachmultiple-choicequestion,weconstructasinglechat-formatteduser
638
messagewithasinglevideoblockfollowedbyasingletextblock. Thetextusesthefixedtemplate
639
640 You are watching a short first-person (egocentric) video clip.
641 Question: {q}
642
643 {options}
644
645 Answer with only the letter (A, B, C, or D) of the correct option.
where{options}liststhefourcandidateanswersasA.·/B.·/C.·/D.·,oneperline. Frames
646
are passed via Qwen’s chat template with explicit resized_height = resized_width = 224.
647
Predictionsareobtainedfromasingleforwardpass: wereadthenext-tokenlogitsatthelastprompt
648
positionandtakethe overthefourtokenIDscorrespondingtotheletters{A,B,C,D}. Weusethe
649
single-tokenencodingsof“A”,“B”,“C”,and“D”obtainedwithadd_special_tokens=False,and
650
verifythatallfouranswerlettersarerepresentedassingletokensundertheQwen2.5-VLtokenizer.
651
Nogenerationisperformed,andnolengthnormalizationisapplied,sothetaskistreatedasfour-way
652
classificationontopoftheLMhead.
653
Framesamplingandvisual-tokencounts. EachclipisuniformlysampledintoT =128RGB
654 vlm
framesfortheVLMinputandT =128framesforthetrajectoryencoder. Framesareresizedto
655 traj
2242. With Qwen2.5-VL’s spatial-merge size 2, temporal patch size 2, and patch size 14, every
656
2242framecontributes(224/14/2)2=64spatialtokens. The128framesarefoldedintoT =64
657 merged
temporalgroups,giving
658
N = T ·n = 64×64 = 4,096. (10)
merged spatial
Thetextpromptcontributesroughly80to110tokens,sothefull-resolutionsequencelengthfedto
659
theLLMisL ≈4,200.
660 full
Tokenreductionandcomputesavings. Letρ∈(0,1)denotethemergeratio,i.e.,thefractionof
661
visualtokensremoved,andletR=⌊ρN⌋bethenumberofremovedtokens. Bipartitegaze-weighted
662
mergingshortenstheLLMinputfromL toL ≈(1−ρ)N +L . Becausethevisualencoderis
663 full ρ text
frozenandtokenmergingoperatesbetweentheViTandtheLLM,ViTFLOPsareunchanged,and
664
thesavingsaccruetothelanguagemodel. PerLLMlayer,self-attentionscalesasO(L2d)andthe
665
MLPasO(Ld2),whered=3584forQwen2.5-VL-7B.Table5reportstheestimatedper-layerFLOP
666
reduction.
667
Atourdefaultρ=0.90,theestimatedLLMFLOPsarereducedby∼15.6×. Theshortenedsequence
668
alsoreducesattention,memory,andKV-cachesize. Qwen2.5-VL-7Busesgrouped-queryattention
669
with4KVheadsandheaddimension128,givingaKVfeaturedimensionof4×128=512perlayer.
670
Afull-contextcacherequires2·28·4,200·512·2B≈241MBinbfloat16,whereasatρ=0.90it
671
isapproximately29MB.
672
Scorealignment. Thetrajectoryencoderproducesper-frameimportancemapsofshape(T ,196),
673 traj
corresponding to a 14×14 patch grid at each of the T =128 trajectory frames. These maps are
674 traj
aligned to the N=4,096 visual tokens exposed by Qwen2.5-VL to the LLM. The alignment has
675
16

Table5: LLMcomputeasafunctionofthemergeratioρ. Per-layerFLOPscombineself-attention
(∝L2d)andMLP(∝Ld2);ViTandtrajectory-encodercostsareunchangedandthereforeexcluded
from the reduction factor. The reduction factor is relative to the no-merge baseline. FLOPs are
estimatedfortheLLMtraining-modeforwardandbackwardcomputation.
Setting Keepratio Kepttokens LLMseq.lengthL FLOPsreduction
Full(nomerge) 100% 4,096 ∼4,200 1.0×
ρ=0.90 10% 410 ∼510 15.6×
ρ=0.95 5% 205 ∼305 27.5×
ρ=0.97 3% 123 ∼223 39.6×
spatialandtemporalsteps. Spatially,each14×14mapisnearest-neighborupsampledto16×16and
676
thenaverage-pooledwitha2×2kernelto8×8,yieldingn =64scoresperframe,matchingthe64
677 spatial
spatialtokensproducedbyQwen’sspatialmerge. Theintermediateupsamplingisusedbecausedirect
678
poolingfrom14×14withstride2wouldproducea7×7grid,whichdoesnotmatchtheVLMtoken
679
layout. Temporally,theresulting(T ,64)scoretensorislinearlyinterpolatedtotheT =64
680 traj merged
temporalgroupsproducedbyQwen’stemporalmerge. Thealignedtensorisflattenedtos ∈ RN,
681
givingoneimportancescorepervisualtoken.
682
Sequence construction with merged tokens. Given the aligned per-token importance scores
683
s∈RN,thetop(N−R)positionsbyscorearedesignatedasreceiversandthebottomRpositionsas
684
sources,whereR=⌊ρN⌋. Eachsourceismatchedtothereceiverwiththehighestcosinesimilarity
685
intoken-embeddingspace. Forareceivertokeni,letS(i)denotethesetofsourcetokensassignedto
686
it. Themergedembeddingiscomputedasascore-weightedaverage,
687
(cid:80)
s v + s v
i i j∈S(i) j j
v˜ = , (11)
i s + (cid:80) s +ϵ
i j∈S(i) j
implemented with scatter_add. Source positions are then removed from the input sequence
688
ratherthanmasked: thecorrespondingentriesofinput_ids,attention_mask,andthe3D-RoPE
689
position_idsareslicedout,andtheremainingreceiverpositionsarepopulatedwiththemergedem-
690
beddingsviainputs_embeds. Bothtrainingandinferencethereforeoperateattheshortenedlength
691
L ,sotheLLM-sidesavingsinTable5applytobothforwardandbackwardpasses. Thediscrete
692 ρ
top-k receiver/sourceassignmentandnearest-neighbormatchingaretreatedasnon-differentiable
693
routing operations, while gradients flow through the score-weighted averaging operation to the
694
selectedtokenscoresandthetrajectoryencoder.
695
Trainingprocedure. WetrainStage2for3epochsovertheunionofEgoExoLearnandHoloAssist
696
(5,799itemsin8MCQtasks)andreportonEGTEA[18](526items).OptimizationusesAdamW[57]
697
withweightdecay10−4,learningrate1×10−4fortheLoRAadaptersand1×10−5forthetrajectory
698
encoder,aconstantschedulewithoutwarmup,andglobalgradient-normclippingat1.0. Theper-
699
device batch size is fixed at 1, and gradients are accumulated over 4 steps, yielding an effective
700
batchsizeof4. Stage2runsonasingleNVIDIAH200GPUinbfloat16. Hyperparametersare
701
summarizedinTable6.
702
Trainingobjective. Letℓstu ∈R|V|denotethelast-positionLLMlogitsoverthemergedsequence
703
andlety⋆ ∈ {A,B,C,D}bethegoldanswerletter. Werestrictthelogitstothefouroption-letter
704
tokenIDs,ℓstu ∈R4,andapplycross-entropy:
705 C
exp (cid:0) ℓstu[y⋆] (cid:1)
L = −log C . (12)
CE (cid:80) exp (cid:0) ℓstu[c] (cid:1)
c∈C C
WeuseL=L asthesoletrainingobjective: thereisnoteacherpass,logitdistillation,orauxiliary
706 CE
KDterm. Thisgivesasingle-forward-per-steptrainingloop. Gradientsarebackpropagatedthrough
707
thedifferentiablescore-weightedmergeoperationtoupdatetheLM-sideLoRAadaptersandthe
708
trajectoryencoder,whilethediscretetoken-routingdecisionsaretreatedasfixedwithineachforward
709
pass.
710
17

Table6: Stage-2traininghyperparameters.
| Component | Hyperparameter | Value  |     |
| --------- | -------------- | ------ | --- |
|           | Optimizer      | AdamW  |     |
|           | Weightdecay    | 1×10−4 |     |
Optimizer
|          | LoRAlearningrate    | 1×10−4            |     |
| -------- | ------------------- | ----------------- | --- |
|          | TrajectoryencoderLR | 1×10−5            |     |
|          | Schedule            | constant,nowarmup |     |
| Schedule | Epochs              | 3                 |     |
|          | Gradientclipping    | 1.0(globalℓ       | )   |
2
|          | Per-devicebatch      | 1         |           |
| -------- | -------------------- | --------- | --------- |
| Batching | Gradientaccumulation | 4         |           |
|          | Effectivebatch       | 4         |           |
|          | Targets              | {q,k,v,o} | inLLMonly |
proj
| LoRA | Rank/α/dropout  | 16/32/0.05         |     |
| ---- | --------------- | ------------------ | --- |
|      | Trainableparams | 10,092,544(0.122%) |     |
|      | VLMframesT      | 128                |     |
vlm
|     | TrajectoryframesT | 128 |     |
| --- | ----------------- | --- | --- |
Inputs traj
|          | Visualkeyframes(DINOv2) | 16                     |     |
| -------- | ----------------------- | ---------------------- | --- |
|          | Frameresolution         | 224×224                |     |
| Hardware | GPU                     | 1×NVIDIAH200(bfloat16) |     |
A.4 BaselineModels
711
712 Rule-basedgaze/handpruner. AsanaivecounterparttoTrajGazeMerge,wedesignaheuristic
713 patchselectorthatreplacesthelearnedTrajGazemodulewitharule-basedpriorityscore. Thescore
fuses(i)per-patchquestion–imagecosinesimilarityfromTalk2DINO[58]and(ii)Gaussianspatial
714
priorscenteredontheper-framegazefixationandthetwohandlandmarks. Giventhepastegocentric
715
frames, thetextualquestion, andper-framegazeandhand2Dcoordinates, theselectorproduces
716
717 aTrajGaze-compatiblegazinginformationrecord. Whengazeandhandarebothunavailablefor
718 aframe,thepriorreducestoauniformmap,andtheselectionisdrivenpurelybyquestion–patch
similarity. Retainingthetop10%patches,itservesasadrop-inreplacementforourQwen2.5-VL-7B-
719
Instructpipeline. Weevaluatethisbaselineinbothzero-shotandLoRAfine-tuningsettings,keeping
720
allotherhyperparametersidenticaltoisolatetheeffectofthelearnedselector.
721
722 FastVID[34]. FastVIDisatraining-free,dynamic-densitytokenpruningmethodthatcompresses
723 visiontokensthroughathree-stagecascadeappliedbetweenthevisionencoderandtheLLM:dynamic
segmentation(DySeg)thatpartitionsthevideointotemporallycoherentchunks,spatio-temporal
724
pruning(STPrune)thatretainsthemostsalientandthemostcontext-representativetokenspersegment,
725
anddensetokenmerging(DTM)thataggregatesthediscardedtokensbackintothesurvivingones
726
viaattention-weightedmerging. WechooseFastVIDasadirectcounterparttoours’gaze-weighted
727
728 token-mergemodule,sincebothmethodsreducethenumberofvisiontokensforwardedtotheLLM
729 butrelyondifferentselectionsignals. WeintegrateFastVIDintoQwen2.5-VL-7B-Instructusingthe
authors’forkofthevisiontower,whichexposesthelast-blockattentionmaprequiredbySTPrune.
730
Forafaircomparison,weusethesameframebudgetasinourmodel.
731
AutoGaze[16]. AutoGazeisalearned,multi-scaleautoregressivepatchselectortrainedviareinforce-
732
733 mentlearning,witharewardsignalbasedonreconstructionfidelity. Ateachselectionstepthepolicy
734 attendstopreviouslyselectedpatchesandgreedilypicksthenextmostreconstruction-informative
patch,buildingamulti-scaleselectionschedulethatprogressivelycoversthemostvisuallysalient
735
regions. Becausetrainingisdrivenentirelybypixel-levelreconstructionquality,theselectedpatches
736
tendtofocusonvisuallycomplexorhigh-frequencyregionsratherthanonsemanticallyrelevant
737
738 regionsforadownstreamquestion. AutoGazeservesasadirectlearned-selectioncounterpartto
739 TrajGazeMerge, contrastinga reconstruction-driven RL policy with our question- andtrajectory-
conditioned scoring strategy. We adapt the AutoGaze output into a compatible format for our
740
Qwen2.5-VL-7B-Instructpipelineandevaluateitunderidenticalzero-shotandLoRAfine-tuning
741
conditions,maintainingthe10%visualtokenbudgetandtheoriginalmulti-scalescheduletoisolate
742
theeffectoftheselectioncriterionalone.
743
18

VisionZip[40]. VisionZipisatraining-free,text-agnostictoken-reductionmethodthatcompresses
744
visualtokensintwostagesusingsignalsfromthevisualencoder’sownself-attention. Inthefirst
745
stage,dominanttokensareselectedbyrankingpatchtokensaccordingtotheiraggregateattention
746
weightreceivedfromallothertokensinthefinalencoderlayer,retainingthetop-K patchesthatare
747 d
mostgloballyattended.Inthesecondstage,theremainingnon-dominanttokensarecompressedintoa
748
fixednumberofcontextualtokensbyclusteringthemwithk-meansinkey-vectorspaceandreplacing
749
eachclusterwithitscentroid,preservingspatialcoveragewhilereducingredundancy. Thistwo-stage
750
designcapturesbothlocalsaliency(dominantselection)anduniformspatialcoverage(contextual
751
merging)withoutanytask-specifictraining. BecauseVisionZipreliesexclusivelyonencoder-internal
752
attention statistics and ignores the textual query, it provides a strong text-agnostic baseline that
753
contrastswithourgaze-conditionedimportancescoring. WeintegratetheCLS-freevariantintoour
754
Qwen2.5-VL-7B-Instructpipelineandevaluateitunderzero-shotandLoRAfine-tuningconditions
755
ata10%tokenbudget,maintainingidenticaldatasplitsandpreprocessing.
756
PruneVid[37]. PruneVidisatraining-free,question-awarevideotokenpruningmethodthatreduces
757
visual tokens through three sequential steps: temporal clustering, spatiotemporal merging, and
758
question-drivenpruning. First,videoframesaregroupedintotemporalclustersbyvisualsimilarity,
759
andarepresentativekeyframeisselectedperclustertoeliminateredundanttemporalcontent. Second,
760
within each keyframe, spatially adjacent and visually similar patch tokens are merged to reduce
761
intra-frameredundancywhilepreservingspatiallayout. Third,thecompressedtokensetisfurther
762
prunedusingthecross-attentionweightsbetweenthetextquerytokensandthevisualtokensinside
763
the LLM, retaining only the patches most attended to by the question representation. This text-
764
conditionedfinalstageprovidesquestionrelevancewithoutadditionaltraining,makingPruneVid
765
anaturalcounterparttoTrajGazeMerge: bothmethodsselecttokensconditionedonthequery,but
766
PruneVidusesLLMself-attentionastherelevancesignalwhereasweusegaze-andhand-trajectory
767
scoring. WeadaptPruneVidforQwen2.5-VL-7B-Instructandevaluateitinbothzero-shotandLoRA
768
fine-tuningsettingsata10%tokenretentionratiounderidenticalexperimentalconditionsanddata
769
splits.
770
A.5 Reproducibility
771
Code, configuration files, and exact launch scripts are released alongside the pa-
772
per. The Qwen2.5-VL-7B-Instruct snapshot hash used in all experiments is
773
cc594898137f460bfe9f0759e9844b3ce807cfb5. Stage 1 and Stage 2 each use a single
774
NVIDIAH200GPU;alltrainingrunsuseafixedrandomseed(42)appliedtotorch,torch.cuda,
775
random, and numpy. Stage 2 is launched without distributed-data-parallel wrapping to reduce
776
cross-rankfloating-pointvariabilityandimprovereproducibilityforafixedseed.
777
A.6 BroaderImpact
778
Thisworkadvancesefficientegocentricvideounderstandingbyleveraginggazeandhandtrajectories
779
tocompressvisualcontextforlargevision-languagemodels. Practicalapplicationsincludeassistive
780
technologies for people with motor or cognitive impairments, hands-free AR/VR interfaces, and
781
industrialtrainingsystemswhereunderstandingaworker’sattentionalfocusissafety-critical. By
782
reducing the visual token budget to 10% without sacrificing—and in some settings surpassing—
783
full-token accuracy, the method also lowers the computational cost of deploying VLMs on long
784
egocentricstreams,potentiallydemocratizingaccesstosuchcapabilitiesonedgedeviceswithlimited
785
memoryandpowerbudgets. Ontheriskside,systemsthatinferintentfromgazeandhandmotion
786
raiseprivacyconcerns: continuoustrackingofwhereapersonlooksandwhattheytouchconstitutes
787
sensitivebehavioraldatathatcouldberepurposedforsurveillanceorprofilingifdeployedwithout
788
appropriatesafeguards. Weencouragepractitionersadoptingthisframeworktoimplementon-device
789
processing,minimizedataretention,andobtaininformedconsentwhencollectingorusinggazeand
790
handtrackingsignals.
791
A.7 Limitations
792
TrajGazeMergereliesonthequalityandavailabilityofgazeandhand-trackingsignals. Although
793
missingdetectionsarehandledvialearnablemissingembeddings,systematictrackingfailures—such
794
asgazedriftduringfastheadmotionorhandocclusionduringclosemanipulation—candegrade
795
19

thescorer’sspatialestimates,causingthemodeltofallbackonweakertrajectorycues. Thislimits
796
applicabilitytosettingswherereliableeye-andhand-trackinghardwareisdeployed,excludinglarge
797
portionsofin-the-wildegocentricvideowhereonlyRGBstreamsareavailable.
798
Furthermore,thetwo-stagetrainingpipeline—trajectorypretrainingonbehavioraldatafollowedby
799
jointVLMfine-tuning—requirescuratedclipswithsynchronizedgazeandhandannotations,which
800
arescarcerandmoreexpensivetocollectthanthevideo–QApairsusedbytraining-freecompetitors.
801
Theadditionalannotationrequirementmayrestrictscalingtolargerormorediversedatasetswithout
802
substantialdatacollectioneffort.
803
A conceptuallimitation of trajectory-conditioned scoringis that gaze andhand signals primarily
804
encodewheretheuserattendsoracts;tasksthatdependondetectingwhatischanginginthescene—
805
suchasobjectstatetransitionsoreventboundaries—therefore,receiveonlyindirectguidancefrom
806
ourscoringsignal,irrespectiveofthecompressionbudget.
807
Morebroadly,themodelisprimarilyevaluatedattwovisual-tokenbudgets(5%and10%)onasingle
808
egocentric VQA benchmark (StreamGaze [45]), whose source datasets (EGTEA, EgoExoLearn,
809
HoloAssist) are dominated by indoor manipulation activities. While we report estimated LLM
810
computeatretentionratiosaslowas3%intheappendix,end-to-endaccuracyatsub-5%budgets,on
811
substantiallylongervideocontexts,orunderthird-personviewpointsremainsunmeasured. Whether
812
thetrajectory-drivenscoringgeneralizestosettingswithoutsustainedhand–objectinteraction—such
813
asoutdoornavigationorpassiveobservation—isalsoanopenquestion.
814
20

NeurIPSPaperChecklist
815
The checklist is designed to encourage best practices for responsible machine learning research,
816
addressingissuesofreproducibility,transparency,researchethics,andsocietalimpact.Donotremove
817
thechecklist: Thepapersnotincludingthechecklistwillbedeskrejected. Thechecklistshould
818
followthereferencesandfollowthe(optional)supplementalmaterial. ThechecklistdoesNOTcount
819
towardsthepagelimit.
820
Pleasereadthechecklistguidelinescarefullyforinformationonhowtoanswerthesequestions. For
821
eachquestioninthechecklist:
822
• Youshouldanswer[Yes],[No],or[N/A].
823
• [N/A] means either that the question is Not Applicable for that particular paper or the
824
relevantinformationisNotAvailable.
825
• Pleaseprovideashort(1–2sentence)justificationrightafteryouranswer(evenfor[N/A]).
826
Thechecklistanswersareanintegralpartofyourpapersubmission. Theyarevisibletothe
827
reviewers,areachairs,seniorareachairs,andethicsreviewers. Youwillalsobeaskedtoincludeit
828
(aftereventualrevisions)withthefinalversionofyourpaper,anditsfinalversionwillbepublished
829
withthepaper.
830
Thereviewersofyourpaperwillbeaskedtousethechecklistasoneofthefactorsintheirevaluation.
831
While [Yes] is generally preferable to [No], it is perfectly acceptable to answer [No] provided a
832
properjustificationisgiven(e.g.,errorbarsarenotreportedbecauseitwouldbetoocomputationally
833
expensive”or“wewereunabletofindthelicenseforthedatasetweused”). Ingeneral,answering
834
[No] or [N/A] is not grounds for rejection. While the questions are phrased in a binary way, we
835
acknowledgethatthetrueanswerisoftenmorenuanced,sopleasejustuseyourbestjudgmentand
836
writeajustificationtoelaborate. Allsupportingevidencecanappeareitherinthemainpaperorthe
837
supplementalmaterial,providedinappendix. Ifyouanswer[Yes]toaquestion,inthejustification
838
pleasepointtothesection(s)whererelatedmaterialforthequestioncanbefound.
839
IMPORTANT,please:
840
• Deletethisinstructionblock,butkeepthesectionheading“NeurIPSPaperChecklist",
841
• Keepthechecklistsubsectionheadings,questions/answersandguidelinesbelow.
842
• Donotmodifythequestionsandonlyusetheprovidedmacrosforyouranswers.
843
1. Claims
844
Question: Dothemainclaimsmadeintheabstractandintroductionaccuratelyreflectthe
845
paper’scontributionsandscope?
846
Answer: [Yes]
847
Justification: TheabstractandintroductionclearlyoutlinetheproposedTrajGazeMerge
848
framework, its two-stage training process, and the specific performance improvements
849
achievedontheStreamGazebenchmark.
850
Guidelines:
851
• Theanswer[N/A]meansthattheabstractandintroductiondonotincludetheclaims
852
madeinthepaper.
853
• Theabstractand/orintroductionshouldclearlystatetheclaimsmade,includingthe
854
contributionsmadeinthepaperandimportantassumptionsandlimitations. A[No]or
855
[N/A]answertothisquestionwillnotbeperceivedwellbythereviewers.
856
• Theclaimsmadeshouldmatchtheoreticalandexperimentalresults,andreflecthow
857
muchtheresultscanbeexpectedtogeneralizetoothersettings.
858
• Itisfinetoincludeaspirationalgoalsasmotivationaslongasitisclearthatthesegoals
859
arenotattainedbythepaper.
860
2. Limitations
861
Question: Doesthepaperdiscussthelimitationsoftheworkperformedbytheauthors?
862
Answer: [Yes]
863
21

Justification: SectionA.7explicitlydetailsthemethod’slimitations.
864
Guidelines:
865
• Theanswer[N/A]meansthatthepaperhasnolimitationwhiletheanswer[No]means
866
thatthepaperhaslimitations,butthosearenotdiscussedinthepaper.
867
• Theauthorsareencouragedtocreateaseparate“Limitations”sectionintheirpaper.
868
• Thepapershouldpointoutanystrongassumptionsandhowrobusttheresultsareto
869
violationsoftheseassumptions(e.g.,independenceassumptions,noiselesssettings,
870
modelwell-specification,asymptoticapproximationsonlyholdinglocally).Theauthors
871
shouldreflectonhowtheseassumptionsmightbeviolatedinpracticeandwhatthe
872
implicationswouldbe.
873
• Theauthorsshouldreflectonthescopeoftheclaimsmade,e.g.,iftheapproachwas
874
onlytestedonafewdatasetsorwithafewruns. Ingeneral,empiricalresultsoften
875
dependonimplicitassumptions,whichshouldbearticulated.
876
• Theauthorsshouldreflectonthefactorsthatinfluencetheperformanceoftheapproach.
877
Forexample,afacialrecognitionalgorithmmayperformpoorlywhenimageresolution
878
isloworimagesaretakeninlowlighting. Oraspeech-to-textsystemmightnotbe
879
usedreliablytoprovideclosedcaptionsforonlinelecturesbecauseitfailstohandle
880
technicaljargon.
881
• Theauthorsshoulddiscussthecomputationalefficiencyoftheproposedalgorithms
882
andhowtheyscalewithdatasetsize.
883
• If applicable, the authors should discuss possible limitations of their approach to
884
addressproblemsofprivacyandfairness.
885
• Whiletheauthorsmightfearthatcompletehonestyaboutlimitationsmightbeusedby
886
reviewersasgroundsforrejection,aworseoutcomemightbethatreviewersdiscover
887
limitationsthataren’tacknowledgedinthepaper. Theauthorsshouldusetheirbest
888
judgmentandrecognizethatindividualactionsinfavoroftransparencyplayanimpor-
889
tantroleindevelopingnormsthatpreservetheintegrityofthecommunity. Reviewers
890
willbespecificallyinstructedtonotpenalizehonestyconcerninglimitations.
891
3. Theoryassumptionsandproofs
892
Question: Foreachtheoreticalresult,doesthepaperprovidethefullsetofassumptionsand
893
acomplete(andcorrect)proof?
894
Answer: [N/A]
895
Justification: Thepaperdoesnotintroducenewtheoreticalclaimsormathematicalproofs.
896
Guidelines:
897
• Theanswer[N/A]meansthatthepaperdoesnotincludetheoreticalresults.
898
• Allthetheorems, formulas, andproofsinthepapershouldbenumberedandcross-
899
referenced.
900
• Allassumptionsshouldbeclearlystatedorreferencedinthestatementofanytheorems.
901
• Theproofscaneitherappearinthemainpaperorthesupplementalmaterial, butif
902
theyappearinthesupplementalmaterial,theauthorsareencouragedtoprovideashort
903
proofsketchtoprovideintuition.
904
• Inversely,anyinformalproofprovidedinthecoreofthepapershouldbecomplemented
905
byformalproofsprovidedinappendixorsupplementalmaterial.
906
• TheoremsandLemmasthattheproofreliesuponshouldbeproperlyreferenced.
907
4. Experimentalresultreproducibility
908
Question: Doesthepaperfullydisclosealltheinformationneededtoreproducethemainex-
909
perimentalresultsofthepapertotheextentthatitaffectsthemainclaimsand/orconclusions
910
ofthepaper(regardlessofwhetherthecodeanddataareprovidedornot)?
911
Answer: [Yes]
912
Justification: AppendixAprovidescomprehensiveimplementationdetails,includingmodel
913
architectures,hyperparameters,datasplits,andthebasemodelsnapshothash.
914
Guidelines:
915
22

• Theanswer[N/A]meansthatthepaperdoesnotincludeexperiments.
916
• Ifthepaperincludesexperiments,a[No]answertothisquestionwillnotbeperceived
917
well by the reviewers: Making the paper reproducible is important, regardless of
918
whetherthecodeanddataareprovidedornot.
919
• Ifthecontributionisadatasetand/ormodel,theauthorsshoulddescribethestepstaken
920
tomaketheirresultsreproducibleorverifiable.
921
• Dependingonthecontribution,reproducibilitycanbeaccomplishedinvariousways.
922
Forexample,ifthecontributionisanovelarchitecture,describingthearchitecturefully
923
mightsuffice,orifthecontributionisaspecificmodelandempiricalevaluation,itmay
924
benecessarytoeithermakeitpossibleforotherstoreplicatethemodelwiththesame
925
dataset,orprovideaccesstothemodel. Ingeneral. releasingcodeanddataisoften
926
onegoodwaytoaccomplishthis,butreproducibilitycanalsobeprovidedviadetailed
927
instructionsforhowtoreplicatetheresults,accesstoahostedmodel(e.g.,inthecase
928
ofalargelanguagemodel),releasingofamodelcheckpoint,orothermeansthatare
929
appropriatetotheresearchperformed.
930
• WhileNeurIPSdoesnotrequirereleasingcode,theconferencedoesrequireallsubmis-
931
sionstoprovidesomereasonableavenueforreproducibility,whichmaydependonthe
932
natureofthecontribution. Forexample
933
(a) Ifthecontributionisprimarilyanewalgorithm,thepapershouldmakeitclearhow
934
toreproducethatalgorithm.
935
(b) Ifthecontributionisprimarilyanewmodelarchitecture,thepapershoulddescribe
936
thearchitectureclearlyandfully.
937
(c) Ifthecontributionisanewmodel(e.g.,alargelanguagemodel),thenthereshould
938
eitherbeawaytoaccessthismodelforreproducingtheresultsorawaytoreproduce
939
themodel(e.g.,withanopen-sourcedatasetorinstructionsforhowtoconstruct
940
thedataset).
941
(d) We recognize that reproducibility may be tricky in some cases, in which case
942
authorsarewelcometodescribetheparticularwaytheyprovideforreproducibility.
943
Inthecaseofclosed-sourcemodels,itmaybethataccesstothemodelislimitedin
944
someway(e.g.,toregisteredusers),butitshouldbepossibleforotherresearchers
945
tohavesomepathtoreproducingorverifyingtheresults.
946
5. Openaccesstodataandcode
947
Question: Doesthepaperprovideopenaccesstothedataandcode,withsufficientinstruc-
948
tionstofaithfullyreproducethemainexperimentalresults,asdescribedinsupplemental
949
material?
950
Answer: [No]
951
Justification: Wedonotprovideatthetimeofsubmission,butwewillprovideopenaccess
952
tothecode.
953
Guidelines:
954
• Theanswer[N/A]meansthatpaperdoesnotincludeexperimentsrequiringcode.
955
• PleaseseetheNeurIPScodeanddatasubmissionguidelines(https://neurips.cc/
956
public/guides/CodeSubmissionPolicy)formoredetails.
957
• Whileweencouragethereleaseofcodeanddata,weunderstandthatthismightnot
958
bepossible,so[No]isanacceptableanswer. Paperscannotberejectedsimplyfornot
959
includingcode,unlessthisiscentraltothecontribution(e.g.,foranewopen-source
960
benchmark).
961
• Theinstructionsshouldcontaintheexactcommandandenvironmentneededtorunto
962
reproducetheresults. SeetheNeurIPScodeanddatasubmissionguidelines(https:
963
//neurips.cc/public/guides/CodeSubmissionPolicy)formoredetails.
964
• Theauthorsshouldprovideinstructionsondataaccessandpreparation,includinghow
965
toaccesstherawdata,preprocesseddata,intermediatedata,andgenerateddata,etc.
966
• Theauthorsshouldprovidescriptstoreproduceallexperimentalresultsforthenew
967
proposedmethodandbaselines. Ifonlyasubsetofexperimentsarereproducible,they
968
shouldstatewhichonesareomittedfromthescriptandwhy.
969
23

• Atsubmissiontime, topreserveanonymity, theauthorsshouldreleaseanonymized
970
versions(ifapplicable).
971
• Providingasmuchinformationaspossibleinsupplementalmaterial(appendedtothe
972
paper)isrecommended,butincludingURLstodataandcodeispermitted.
973
6. Experimentalsetting/details
974
Question: Doesthepaperspecifyallthetrainingandtestdetails(e.g.,datasplits,hyperpa-
975
rameters,howtheywerechosen,typeofoptimizer)necessarytounderstandtheresults?
976
Answer: [Yes]
977
Justification: Section4andAppendixAthoroughlydocumenttheevaluationprotocol,task
978
definitions,tokenbudgets,andoptimizationsettings.
979
Guidelines:
980
• Theanswer[N/A]meansthatthepaperdoesnotincludeexperiments.
981
• Theexperimentalsettingshouldbepresentedinthecoreofthepapertoalevelofdetail
982
thatisnecessarytoappreciatetheresultsandmakesenseofthem.
983
• Thefulldetailscanbeprovidedeitherwiththecode,inappendix,orassupplemental
984
material.
985
7. Experimentstatisticalsignificance
986
Question:Doesthepaperreporterrorbarssuitablyandcorrectlydefinedorotherappropriate
987
informationaboutthestatisticalsignificanceoftheexperiments?
988
Answer: [No]
989
Justification: Wereportabsoluteaccuracymetricswithouterrorbars,whichisastandard
990
practicewhenevaluatinglarge-scaleMLLMs.
991
Guidelines:
992
• Theanswer[N/A]meansthatthepaperdoesnotincludeexperiments.
993
• Theauthorsshouldanswer[Yes]iftheresultsareaccompaniedbyerrorbars,confidence
994
intervals,orstatisticalsignificancetests,atleastfortheexperimentsthatsupportthe
995
mainclaimsofthepaper.
996
• Thefactorsofvariabilitythattheerrorbarsarecapturingshouldbeclearlystated(for
997
example,train/testsplit,initialization,randomdrawingofsomeparameter,oroverall
998
runwithgivenexperimentalconditions).
999
• Themethodforcalculatingtheerrorbarsshouldbeexplained(closedformformula,
1000
calltoalibraryfunction,bootstrap,etc.)
1001
• Theassumptionsmadeshouldbegiven(e.g.,Normallydistributederrors).
1002
• Itshouldbeclearwhethertheerrorbaristhestandarddeviationorthestandarderror
1003
ofthemean.
1004
• It is OK to report 1-sigma error bars, but one should state it. The authors should
1005
preferablyreporta2-sigmaerrorbarthanstatethattheyhavea96%CI,ifthehypothesis
1006
ofNormalityoferrorsisnotverified.
1007
• Forasymmetricdistributions,theauthorsshouldbecarefulnottoshowintablesor
1008
figuressymmetricerrorbarsthatwouldyieldresultsthatareoutofrange(e.g.,negative
1009
errorrates).
1010
• Iferrorbarsarereportedintablesorplots,theauthorsshouldexplaininthetexthow
1011
theywerecalculatedandreferencethecorrespondingfiguresortablesinthetext.
1012
8. Experimentscomputeresources
1013
Question: Foreachexperiment,doesthepaperprovidesufficientinformationonthecom-
1014
puterresources(typeofcomputeworkers,memory,timeofexecution)neededtoreproduce
1015
theexperiments?
1016
Answer: [Yes]
1017
Justification: Appendix A specifies the use of resources and details the exact training
1018
durationandestimatedFLOPreductions.
1019
Guidelines:
1020
24

• Theanswer[N/A]meansthatthepaperdoesnotincludeexperiments.
1021
• ThepapershouldindicatethetypeofcomputeworkersCPUorGPU,internalcluster,
1022
orcloudprovider,includingrelevantmemoryandstorage.
1023
• Thepapershouldprovidetheamountofcomputerequiredforeachoftheindividual
1024
experimentalrunsaswellasestimatethetotalcompute.
1025
• Thepapershoulddisclosewhetherthefullresearchprojectrequiredmorecompute
1026
thantheexperimentsreportedinthepaper(e.g.,preliminaryorfailedexperimentsthat
1027
didn’tmakeitintothepaper).
1028
9. Codeofethics
1029
Question: Doestheresearchconductedinthepaperconform, ineveryrespect, withthe
1030
NeurIPSCodeofEthicshttps://neurips.cc/public/EthicsGuidelines?
1031
Answer: [Yes]
1032
Justification: Weutilizesexisting,publicdatasetsandstandardmethodologies,adhering
1033
strictlytoestablishedethicalguidelinesinmachinelearning.
1034
Guidelines:
1035
• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of
1036
Ethics.
1037
• Iftheauthorsanswer[No],theyshouldexplainthespecialcircumstancesthatrequirea
1038
deviationfromtheCodeofEthics.
1039
• Theauthorsshouldmakesuretopreserveanonymity(e.g.,ifthereisaspecialconsid-
1040
erationduetolawsorregulationsintheirjurisdiction).
1041
10. Broaderimpacts
1042
Question: Does the paper discuss both potential positive societal impacts and negative
1043
societalimpactsoftheworkperformed?
1044
Answer: [Yes]
1045
Justification: SectionA.6detailsthemethod’sbroaderimpact.
1046
Guidelines:
1047
• Theanswer[N/A]meansthatthereisnosocietalimpactoftheworkperformed.
1048
• Iftheauthorsanswer[N/A]or[No],theyshouldexplainwhytheirworkhasnosocietal
1049
impactorwhythepaperdoesnotaddresssocietalimpact.
1050
• Examplesofnegativesocietalimpactsincludepotentialmaliciousorunintendeduses
1051
(e.g.,disinformation,generatingfakeprofiles,surveillance),fairnessconsiderations
1052
(e.g.,deploymentoftechnologiesthatcouldmakedecisionsthatunfairlyimpactspecific
1053
groups),privacyconsiderations,andsecurityconsiderations.
1054
• Theconferenceexpectsthatmanypaperswillbefoundationalresearchandnottied
1055
toparticularapplications,letalonedeployments. However,ifthereisadirectpathto
1056
anynegativeapplications,theauthorsshouldpointitout. Forexample,itislegitimate
1057
topointoutthatanimprovementinthequalityofgenerativemodelscouldbeusedto
1058
generateDeepfakesfordisinformation. Ontheotherhand,itisnotneededtopointout
1059
thatagenericalgorithmforoptimizingneuralnetworkscouldenablepeopletotrain
1060
modelsthatgenerateDeepfakesfaster.
1061
• Theauthorsshouldconsiderpossibleharmsthatcouldarisewhenthetechnologyis
1062
being used as intended and functioning correctly, harms that could arise when the
1063
technologyisbeingusedasintendedbutgivesincorrectresults,andharmsfollowing
1064
from(intentionalorunintentional)misuseofthetechnology.
1065
• Iftherearenegativesocietalimpacts,theauthorscouldalsodiscusspossiblemitigation
1066
strategies (e.g., gated release of models, providing defenses in addition to attacks,
1067
mechanismsformonitoringmisuse,mechanismstomonitorhowasystemlearnsfrom
1068
feedbackovertime,improvingtheefficiencyandaccessibilityofML).
1069
11. Safeguards
1070
Question: Doesthepaperdescribesafeguardsthathavebeenputinplaceforresponsible
1071
releaseofdataormodelsthathaveahighriskformisuse(e.g.,pre-trainedlanguagemodels,
1072
imagegenerators,orscrapeddatasets)?
1073
25

Answer: [N/A]
1074
Justification: Thecontributionisaspecifictokencompressionmoduleforegocentricvideo,
1075
whichdoesnotposethehighmisuserisksassociatedwithfoundationalgenerativemodels
1076
ornewlyscrapeddatasets.
1077
Guidelines:
1078
• Theanswer[N/A]meansthatthepaperposesnosuchrisks.
1079
• Releasedmodelsthathaveahighriskformisuseordual-useshouldbereleasedwith
1080
necessarysafeguardstoallowforcontrolleduseofthemodel,forexamplebyrequiring
1081
thatusersadheretousageguidelinesorrestrictionstoaccessthemodelorimplementing
1082
safetyfilters.
1083
• DatasetsthathavebeenscrapedfromtheInternetcouldposesafetyrisks. Theauthors
1084
shoulddescribehowtheyavoidedreleasingunsafeimages.
1085
• Werecognizethatprovidingeffectivesafeguardsischallenging,andmanypapersdo
1086
notrequirethis,butweencourageauthorstotakethisintoaccountandmakeabest
1087
faitheffort.
1088
12. Licensesforexistingassets
1089
Question: Arethecreatorsororiginalownersofassets(e.g.,code,data,models),usedin
1090
thepaper,properlycreditedandarethelicenseandtermsofuseexplicitlymentionedand
1091
properlyrespected?
1092
Answer: [Yes]
1093
Justification: Weproperlycitesallutilizeddatasetsandthechosenvision-languageback-
1094
bone.
1095
Guidelines:
1096
• Theanswer[N/A]meansthatthepaperdoesnotuseexistingassets.
1097
• Theauthorsshouldcitetheoriginalpaperthatproducedthecodepackageordataset.
1098
• Theauthorsshouldstatewhichversionoftheassetisusedand,ifpossible,includea
1099
URL.
1100
• Thenameofthelicense(e.g.,CC-BY4.0)shouldbeincludedforeachasset.
1101
• Forscrapeddatafromaparticularsource(e.g.,website),thecopyrightandtermsof
1102
serviceofthatsourceshouldbeprovided.
1103
• If assets are released, the license, copyright information, and terms of use in the
1104
packageshouldbeprovided. Forpopulardatasets,paperswithcode.com/datasets
1105
hascuratedlicensesforsomedatasets. Theirlicensingguidecanhelpdeterminethe
1106
licenseofadataset.
1107
• Forexistingdatasetsthatarere-packaged,boththeoriginallicenseandthelicenseof
1108
thederivedasset(ifithaschanged)shouldbeprovided.
1109
• Ifthisinformationisnotavailableonline,theauthorsareencouragedtoreachoutto
1110
theasset’screators.
1111
13. Newassets
1112
Question:Arenewassetsintroducedinthepaperwelldocumentedandisthedocumentation
1113
providedalongsidetheassets?
1114
Answer: [No]
1115
Justification: Code,configurationfiles,andlaunchscriptswillbereleasedalongsidethe
1116
paper.
1117
Guidelines:
1118
• Theanswer[N/A]meansthatthepaperdoesnotreleasenewassets.
1119
• Researchersshouldcommunicatethedetailsofthedataset/code/modelaspartoftheir
1120
submissions via structured templates. This includes details about training, license,
1121
limitations,etc.
1122
• Thepapershoulddiscusswhetherandhowconsentwasobtainedfrompeoplewhose
1123
assetisused.
1124
26

• Atsubmissiontime,remembertoanonymizeyourassets(ifapplicable). Youcaneither
1125
createananonymizedURLorincludeananonymizedzipfile.
1126
14. Crowdsourcingandresearchwithhumansubjects
1127
Question: Forcrowdsourcingexperimentsandresearchwithhumansubjects,doesthepaper
1128
includethefulltextofinstructionsgiventoparticipantsandscreenshots,ifapplicable,as
1129
wellasdetailsaboutcompensation(ifany)?
1130
Answer: [N/A]
1131
Justification:TrajGazeMergedidnotinvolveanynewcrowdsourcingordirecthumansubject
1132
datacollection.
1133
Guidelines:
1134
• Theanswer[N/A]meansthatthepaperdoesnotinvolvecrowdsourcingnorresearch
1135
withhumansubjects.
1136
• Includingthisinformationinthesupplementalmaterialisfine,butifthemaincontribu-
1137
tionofthepaperinvolveshumansubjects,thenasmuchdetailaspossibleshouldbe
1138
includedinthemainpaper.
1139
• AccordingtotheNeurIPSCodeofEthics,workersinvolvedindatacollection,curation,
1140
orotherlaborshouldbepaidatleasttheminimumwageinthecountryofthedata
1141
collector.
1142
15. Institutional review board (IRB) approvals or equivalent for research with human
1143
subjects
1144
Question: Doesthepaperdescribepotentialrisksincurredbystudyparticipants,whether
1145
suchrisksweredisclosedtothesubjects,andwhetherInstitutionalReviewBoard(IRB)
1146
approvals(oranequivalentapproval/reviewbasedontherequirementsofyourcountryor
1147
institution)wereobtained?
1148
Answer: [N/A]
1149
Justification: Nonewhumansubjectresearchwasconducted.
1150
Guidelines:
1151
• Theanswer[N/A]meansthatthepaperdoesnotinvolvecrowdsourcingnorresearch
1152
withhumansubjects.
1153
• Dependingonthecountryinwhichresearchisconducted,IRBapproval(orequivalent)
1154
mayberequiredforanyhumansubjectsresearch. IfyouobtainedIRBapproval,you
1155
shouldclearlystatethisinthepaper.
1156
• Werecognizethattheproceduresforthismayvarysignificantlybetweeninstitutions
1157
andlocations,andweexpectauthorstoadheretotheNeurIPSCodeofEthicsandthe
1158
guidelinesfortheirinstitution.
1159
• Forinitialsubmissions,donotincludeanyinformationthatwouldbreakanonymity(if
1160
applicable),suchastheinstitutionconductingthereview.
1161
16. DeclarationofLLMusage
1162
Question: Does the paper describe the usage of LLMs if it is an important, original, or
1163
non-standardcomponentofthecoremethodsinthisresearch? NotethatiftheLLMisused
1164
onlyforwriting,editing,orformattingpurposesanddoesnotimpactthecoremethodology,
1165
scientificrigor,ororiginalityoftheresearch,declarationisnotrequired.
1166
Answer: [N/A]
1167
Justification: LargeLanguageModelswerenotutilizedindesigningthemainframeworkor
1168
formulatingthecoremethodologyofthisresearch.
1169
Guidelines:
1170
• Theanswer[N/A]meansthatthecoremethoddevelopmentinthisresearchdoesnot
1171
involveLLMsasanyimportant,original,ornon-standardcomponents.
1172
• PleaserefertoourLLMpolicyintheNeurIPShandbookforwhatshouldorshouldnot
1173
bedescribed.
1174
27
