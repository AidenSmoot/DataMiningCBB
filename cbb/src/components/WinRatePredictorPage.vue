<template>
  <v-app id="baseWR">
    <v-container class="containerWR">
      <v-card id="cardWR">
        <v-row id="parentRowWR">
          <v-col>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(0) v-model="efgo">

              </v-text-field>
            </v-row>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(1) v-model="efgd">

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(2) v-model="tor">

              </v-text-field>              
            </v-row>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(3) v-model="tord">

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(4) v-model="orb">

              </v-text-field>              
            </v-row>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(5) v-model="drb">

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(6) v-model="ftr">

              </v-text-field>              
            </v-row>
            <v-row class="rowWR">
              <v-text-field :label=getStatName(7) v-model="ftrd">

              </v-text-field>
            </v-row>
          </v-col>
          <v-col id="buttonColWR">
            <v-row class="rowWR" id="buttonRowWR" justify='center'>
              <v-btn id="actionButtonWR" color="red" rounded="xl" @click="logReg()">
              GO
              </v-btn>
            </v-row>
          </v-col>          
        </v-row>
        <v-divider/>
        <v-row id="algorithmsWR">
          <v-col class="algoContentWR">
            <h1 class="headersWR">Logistic Regression</h1>
            <h3>Overall Predicted Win Rate:</h3>
            <zingchart :data="chartDataLog"></zingchart>
          </v-col>
        <v-divider vertical/>
          <v-col class="algoContentWR">
            <h1 class="headersWR">Stochastic Gradient Descent</h1>
            <h3>Overall Predicted Win Rate:</h3>
            <zingchart :data="chartDataSGD"></zingchart>
          </v-col>
        </v-row>
      </v-card>
    </v-container>
  </v-app>
</template>

<script>
import ZingChart from 'zingchart-vue';
  export default {
    components: {
      'zingchart': ZingChart,
    },
    data: () => ({
      inputStats:
      [
      "EFG_O",
      "EFG_D",
      "TOR",
      "TORD",
      "ORB",
      "DRB",
      "FTR",
      "FTRD"
      ],
      chartDataLog: {
          data: 'chartDataLog',
          type: 'line',
          series:[{
            values: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
          }]
        },
        chartDataSGD: {
          data: 'chartDataSGD',
          type: 'line',
          series:[{
            values: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
          }]
        },
      efgo: 0.0,
      efgd: 0.0,
      tor: 0.0,
      tord: 0.0,
      orb: 0.0,
      drb: 0.0,
      ftr: 0.0,
      ftrd: 0.0,
      predWRLog: 0.0,
      predWRSGD: 0.0,
      overallWR: 0.0,
      overallWeights: 
      [
        .06996,
        .37081,
        -.29277,
        -.21185,
        .26236,
        .17328,
        -.17444,
        .07349,
        -.10975
      ],
      overallMeans:
      [
      50.33581,
      50.53196,
      18.24821,
      18.17548,
      28.37521,
      28.55675,
      31.59559,
      31.79366
    ],
    overallSTDV:
    [
      2.69268,
      2.67978,
      1.98619,
      2.29973,
      3.98838,
      2.91061,
      4.48446,
      5.30067
    ],
    annualWeights: //For Each Year
    [ //Bias,               EFG_O,                EFG_D,                TOR,                  TORD,               ORB,                DRB,                  FTR,                FTRD
    [-0.0553811710450956, 0.3816577511749285, -0.3400259351920352, -0.23885389981218574, 0.2555421582602129, 0.15834319478068815, -0.11080183287585126, 0.12871332091561782, -0.07496376209593747], //2013
    [-0.042235813828167906, 0.3468741640136742, -0.3110611183777042, -0.21516485834255464, 0.2457958080578609, 0.1943297570205531, -0.1380367552745388, 0.11742677378713844, -0.09771959564746317], //2014
    [-0.04564240581401962, 0.37236622364468747, -0.30454567082312217, -0.25807264516545847, 0.2221223842032869, 0.18630838924049262, -0.12260277855488529, 0.0869574052575868, -0.11027038644317703], //2015
    [-0.03887426860968346, 0.36610119408608777, -0.3169428387867101, -0.1974217952018684, 0.2472391561388644, 0.16831272328345248, -0.1344154127294317, 0.08691389786156045, -0.09555852107219288], //2016
    [0.12836650414638764, 0.3657175165747584, -0.2806121656159695, -0.1924905860241949, 0.22936065518704282, 0.16553624067656203, -0.11463886127018709, 0.09112287480286928, -0.05942895418837742], //2017
    [0.17545408990251568, 0.3658764763865912, -0.31939161558791024, -0.21596595848371064, 0.23390442460148045, 0.14012778507763474, -0.11606437166880369, 0.07430311191099923, -0.05217115831573972], //2018
    [0.12958093668546575, 0.35915592420527215, -0.2831438948984117, -0.22283515611882748, 0.23550704825974578, 0.20271076568068497, -0.16677716996160447, 0.04782537285514372, -0.023109251113469693], //2019
    [0.1563200759342138, 0.4046995621025649, -0.26503087547182896, -0.1917342184352032, 0.235534351875204, 0.2070714370431145, -0.12995643675590918, 0.09090281295933227, -0.08470769408118263], //2020
    [0.13608202748284223, 0.3820813894802935, -0.2848858968439085, -0.27820244609997624, 0.25917540244343623, 0.19747129044730893, -0.13990878255080685, 0.09368487397813408, -0.07036680540607025], //2021
    [0.15733569787073998, 0.3712281339159596, -0.2846956903377171, -0.2401529403582764, 0.24073966812310835, 0.183441106084842, -0.15801511936786447, 0.1179213258401548, -0.022882679634146388], //2022
    [0.15850232214064836, 0.35435326505857506, -0.30465782480494713, -0.20139991581250827, 0.21744304238251447, 0.1807398026303226, -0.09579657972750324, 0.10000639645380767, -0.031137700489969528] //2023
    ],
    annualMeans:
    [
    [48.53602305,	48.7426513,	20.0259366,	19.95475504,	31.59942363,	31.81556196,	35.97060519,	36.22766571], //2013
    [49.48119658,	49.68689459,	18.37179487,	18.28119658,	31.16752137,	31.3988604,	40.47321937,	40.8002849], //2014
    [48.97378917,	49.18062678,	19.12564103,	19.03988604,	30.86296296,	31.09002849,	37.04330484,	37.32678063], //2015
    [49.77720798,	49.97008547,	18.18404558,	18.11994302,	29.61823362,	29.82108262,	36.63817664,	36.93618234], //2016
    [50.38119658,	50.57321937,	18.59401709,	18.54330484,	29.12079772,	29.2985755,	35.32820513,	35.64245014], //2017
    [50.86666667,	51.06296296,	18.44871795,	18.37977208,	28.54358974,	28.71310541,	33.53760684,	33.77094017], //2018
    [50.60084986,	50.77450425,	18.60651558,	18.52351275,	28.24589235,	28.4203966,	32.95439093,	33.20254958], //2019
    [49.56855524,	49.60623229,	18.9203966,	18.89745042,	27.8878187,	27.96798867,	32.64362606,	32.79235127], //2020
    [49.99423631,	50.22420749,	18.97291066,	18.90086455,	27.66887608,	27.9740634,	31.60518732,	31.8870317], //2021
    [49.89748603,	50.09106145,	18.42681564,	18.37150838,	27.95530726,	28.15391061,	30.32067039,	30.61648045], //2022
    [50.33581267,	50.53195592,	18.24820937,	18.17548209,	28.37520661,	28.55674931,	31.59559229,	31.79366391] //2023
    ],
    annualSTDV:
    [
    [3.137520319,	2.9684842,	2.160322429,	2.358759293,	3.808529743,	2.987599716,	4.684169235,	5.736378279], //2013
    [2.848619347,	2.939862448,	2.054742807,	2.061487788,	3.820013926,	2.667958845,	5.234748097,	6.43893737], //2014
    [3.215657412,	2.89711924,	2.067610228,	2.134455898,	3.937408623,	2.908723053,	4.740443007,	6.146578105], //2015
    [3.024551797,	2.683767761,	1.896082823,	2.080831249,	4.103616397,	2.869678396,	4.603748143,	5.871542866], //2016
    [3.048747301,	2.811893879,	1.891213328,	2.09793126,	4.082517161,	2.865289516,	4.486945537,	5.790870478], //2017
    [3.048431477,	2.750104126,	1.908443469,	2.030231894,	3.91701056,	2.994258378,	4.479387292,	5.375480788], //2018
    [2.934463454,	2.748994428,	2.06378116,	2.089752987,	3.932791087,	2.91919172,	4.702044548,	5.072932031], //2019
    [2.789671848,	2.775901093,	2.018114246,	2.254229707,	3.990568775,	2.935659566,	4.865636037,	5.932378469], //2020
    [3.106629739,	2.87572402,	2.255844385,	2.255808858,	4.319550042,	3.169835572,	4.694509228,	5.491956823], //2021
    [2.924338123,	2.765929686,	2.069600783,	2.348177342,	3.975590647,	3.049127325,	4.388725728,	5.264037569], //2022
    [2.692676751,	2.679783158,	1.986168658,	2.299725574,	3.988384257,	2.910610855,	4.484461372,	5.300671881] //2023
    ],
    weightIndexes:
    [
      "BIAS",
      "EFG_O",
      "EFG_D",
      "TOR",
      "TORD",
      "ORB",
      "DRB",
      "FTR",
      "FTRD"
      ],
    years:
    [
      "2013",
      "2014",
      "2015",
      "2016",
      "2017",
      "2018",
      "2019",
      "2020",
      "2021",
      "2022",
      "2023"
    ]
    }),
  mounted() {
    this.logReg()
  },
  methods: {
      getStatName(value) {
        return this.inputStats[value]
      },
      async logReg() {
        let temp = {
          data: 'chartDataLog',
          type: 'line',
          'scale-x': {
            'values': this.years
          },
          series:[{
            values: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
          }]
        }
        this.chartDataLog = temp
        this.chartDataSGD = temp
        let winratesLog = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        let winratesSGD = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for (let i = 0; i < 11; i++) {
          let normal = [((this.efgo-this.annualMeans[i][0])/this.annualSTDV[i][0]),
                    ((this.efgd-this.annualMeans[i][1])/this.annualSTDV[i][1]),
                    ((this.tor-this.annualMeans[i][2])/this.annualSTDV[i][2]),
                    ((this.tord-this.annualMeans[i][3])/this.annualSTDV[i][3]),
                    ((this.orb-this.annualMeans[i][4])/this.annualSTDV[i][4]),
                    ((this.drb-this.annualMeans[i][5])/this.annualSTDV[i][5]),
                    ((this.ftr-this.annualMeans[i][6])/this.annualSTDV[i][6]),
                    ((this.ftrd-this.annualMeans[i][7])/this.annualSTDV[i][7])
          ]
          winratesLog[i] = this.annualWeights[i][0] //Bias term
          winratesSGD[i] = this.annualWeights[i][0] //Bias term
          for (let j = 1; j < 9; j++) {
            winratesLog[i] += (this.annualWeights[i][j] * normal[j-1]) //TODO update with sigmoid
            winratesSGD[i] += (this.annualWeights[i][j] * normal[j-1])
          }
          if (winratesLog[i] > 1.0) {
            winratesLog[i] = 1.0
          }
          else if (winratesLog[i] < 0.0) {
            winratesLog[i] = 0.0
          }
          if (winratesSGD[i] > 1.0) {
            winratesSGD[i] = 1.0
          }
          else if (winratesSGD[i] < 0.0) {
            winratesSGD[i] = 0.0
          }
        }
        this.chartDataLog = {
          data: 'log',
          type: 'line',
          'scale-x': {
            'values': this.years
          },
          series: [
            {
              values: winratesLog,
            }
          ]
        }
        this.chartDataSGD = {
          data: 'log',
          type: 'line',
          'scale-x': {
            'values': this.years
          },
          series: [
            {
              values: winratesSGD,
            }
          ]
        }
        return await this.overall;
      }
    }
  }

</script>

<style>
#baseWR {
  background-color: burlywood;
}
#cardWR {
  min-height: 95%;
  padding: auto;
}
#actionButtonWR {
  padding:auto;
  min-width: 80%;
}
#buttonRowWR {
  align-items: center;
}
#parentRowWR {
  margin: 0px;
}
#algorithmsWR {
  margin: 0px;
}
#log {
  display:block;
  width: 500px;
  height: 500px;
}
.headersWR {
  text-align: center;
}
.algoContentWR {
  min-height: 90%;
  /* text-align: center; */
}
.containerWR {
  margin-top: 10px;
  min-height: 95%;
}
.rowWR {
  padding: 5px;
  margin-top: 0px !important;
}
</style>