<template>
  <v-app>
    <div id="baseHome" class="boxHome">
      <v-col cols="12">
      <v-row id="originalHome">
      <div id="leftHome" class="boxHome">
        <v-container class="containerHome">
          <v-card>
            <h1>Performance Measures</h1>
            <v-row>
                <v-col>
                  Metric
                </v-col>
                <v-col>
                  Logistic Regression
                </v-col>
                <v-col>
                  Stochastic Gradient Descent
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getMetricName(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLogPerf(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDPerf(0)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getMetricName(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLogPerf(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDPerf(1)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getMetricName(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLogPerf(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDPerf(2)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getMetricName(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLogPerf(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDPerf(3)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getMetricName(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLogPerf(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDPerf(4)" readonly/>
                </v-col>
              </v-row>
          </v-card>
        </v-container>
      </div>
      <v-divider vertical/>
      <div id ="rightHome" class="boxHome">
        <v-container id="imageHolderHome" class="containerHome">
          <v-card>
            <h1> Variance Inflation Factors</h1>
            <v-row>
                <v-col>
                  Variable
                </v-col>
                <v-col>
                  Logistic Regression
                </v-col>
                <v-col>
                  Stochastic Gradient Descent
                </v-col>
              </v-row>
            <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(0)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(0)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(1)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(1)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(2)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(2)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(3)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(3)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(4)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(4)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(5)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(5)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(6)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(6)" readonly/>
                </v-col>
              </v-row>
              <v-row>
                <v-col>
                  <v-text-field :model-value="getStatName(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getLinVIF(7)" readonly/>
                </v-col>
                <v-col>
                  <v-text-field :model-value="getSGDVIF(7)" readonly/>
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
      linVIF:
      [
        1.357,
        1.265,
        1.227,
        1.234,
        1.331,
        1.292,
        1.304,
        1.513
      ],
      sgdVIF:
      [
        1.231,
        1.453,
        1.287,
        1.263,
        1.298,
        1.372,
        1.632,
        1.495
      ],
      metrics:
      [
        'Adjusted R2',
        'AIC',
        'BIC',
        'RMSE',
        'Run Time'
      ],
      logisticPerf:
      [
        .886,
        -4452.12,
        -4402.79,
        .0613,
        '.003s'
      ],
      sgdPerf:
      [
        .765,
        -7832.65,
        -7793.74,
        .174,
        '.04s'
      ]
    }),
    methods: {
      getStatName(value) {
        return this.statNames[value]
      },
      getLinVIF(value) {
        return this.linVIF[value]
      },
      getSGDVIF(value) {
        return this.sgdVIF[value]
      },
      getMetricName(value) {
        return this.metrics[value]
      },
      getLogPerf(value) {
        return this.logisticPerf[value]
      },
      getSGDPerf(value) {
        return this.sgdPerf[value]
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