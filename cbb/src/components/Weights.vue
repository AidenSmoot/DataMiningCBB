<template>
  <v-app>
    <div id="baseHome" class="boxHome">
      <v-col cols="12">
      <v-row id="originalHome">
      <div id="leftHome" class="boxHome">
        <v-container class="containerHome">
          <v-card>
            <h1>Logistic Regression Weights</h1>
            <v-row>
                <v-col>
                  Variable
                </v-col>
                <v-col>
                  Standard Deviation
                </v-col>
                <v-col>
                  Change in Winning Odds
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(0)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(1)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(2)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(3)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(4)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(5)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(6)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getPercent(7)" readonly/>
                </v-col>
              </v-row>
          </v-card>
        </v-container>
      </div>
      <v-divider vertical/>
      <div id ="rightHome" class="boxHome">
        <v-container id="imageHolderHome" class="containerHome">
          <v-card>
            <h1> Stochastic Gradient Descent Weights</h1>
            <v-row>
                <v-col>
                  Variable
                </v-col>
                <v-col>
                  Standard Deviation
                </v-col>
                <v-col>
                  Change in Win Rate
                </v-col>
              </v-row>
            <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(0)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(1)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(2)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(3)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(4)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(5)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(6)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSTDEV(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDWeight(7)" readonly/>
                </v-col>
              </v-row>
          </v-card>
        </v-container>
      </div>
      </v-row>
      </v-col>
    </div>
  </v-app>
</template>

<script>
  export default {
    data: () => ({
      statNames:
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
      stdev:
      [
      2.69, 2.68, 1.99, 2.3, 3.99, 2.91, 4.48, 5.3
      ],
      percents:
      [
        45,
        30,
        19,
        8,
        -10,
        -16,
        -19,
        -25
      ],
      sgdoverall:
      [
        0.47688,
        0.071335,
        -0.072819,
        -0.043962,
        0.652954,
        0.396244,
        -0.313865,
        0.0899558,
      ]
    }),
    methods: {
      getStatName(value) {
        return this.statNames[value]
      },
      getSTDEV(value) {
        return this.stdev[value]
      },
      getPercent(value) {
        return this.percents[value] + '%'
      },
      getSGDWeight(value) {
        return this.sgdoverall[value]
      }
  }
}
</script>

<style scoped>
.boxHome {
  min-height: 100%;
}
.containerHome {
  min-height: 100%;
  text-align: center;
}
#originalHome {
  min-height: 100%;
}
#homeImageHome {
  min-width: 100%;
  display: flex;
  margin-left: auto;
  margin-right: auto;
}
.imageHolderHome {
  min-width: 100%;
}
#baseHome {
  background-color:white;
  display: flex;
  min-width: 100%;
}
#contentHome {
  min-height: 100%;
}
#leftHome {
  flex: 1;
  background-color:burlywood;
  max-width: 50%;
}
#rightHome {
  flex: 1;
  background-color: burlywood;
  max-width: 50%;
}
#statsHome {
  background-color: transparent;
}
</style>