<template>
  <v-app>
    <div id="baseHome" class="boxHome">
      <v-col cols="12">
      <v-row id="originalHome">
      <div id="leftHome" class="boxHome">
        <v-container class="containerHome">
          <v-card>
            <h1>Statistics Ranked in Order of Importance</h1>
            <v-divider/>
            <v-list disabled=true class="contentHome">
              <v-list-item id="stats" v-for="n in statsInOrder" :key="n"  :title="n">
              </v-list-item>
            </v-list>
          </v-card>
        </v-container>
      </div>
      <v-divider vertical/>
      <div id ="rightHome" class="boxHome">
        <v-container id="imageHolderHome" class="containerHome">
          <v-card>
            <v-row>
              <v-col cols="8">
                <h1>Feature Weight Over Time</h1>
              </v-col>
              <v-col>
                <v-switch label="Logistic Regression"/>
              </v-col>
            </v-row>
            <v-divider/>
            <img id="homeImageHome" alt="Average Feature Value over Time" src="../assets/LabeledWeightsOverTime.png"/>
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
      statsInOrder:
      [
      "1. EFG_O: Effective Field Goal Percentage Shot",
      "2. EFG_D: Effective Field Goal Percentage Allowed",
      "3. TOR: Turnover Percentage Allowed (Turnover Rate)",
      "4. TORD: Turnover Percentage Committed (Steal Rate)",
      "5. ORB: Offensive Rebound Rate",
      "6. DRB: Offensive Rebound Rate Allowed",
      "7. FTR: Free Throw Rate (How often the given team shoots Free Throws)",
      "8. FTRD: Free Throw Rate Allowed"
      ],
      standardDeviations:
      [
        2.69,
        2.68,
        1.99,
        2.3,
        3.99,
        2.91,
        4.48,
        5.3
      ],
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
      percentIncreases:
      [
        45,
        -25,
        -19,
        30,
        19,
        -16,
        8,
        -10
      ]
    }),
    methods: {
      getStatName(value) {
        return this.statNames[value]
      },
      getPercentIncrease(value) {
        return this.percentIncreases[value] + "%"
      },
      getSTDV(value) {
        return this.standardDeviations[value]
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