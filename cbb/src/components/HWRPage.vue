<template>
  <v-app id="baseHWR">
    <v-container class="containerHWR">
      <v-card id="cardHWR">
        <v-row id="parentRowHWR">
          <v-col>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(0)>

              </v-text-field>
            </v-row>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(1)>

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(2)>

              </v-text-field>              
            </v-row>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(3)>

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(4)>

              </v-text-field>              
            </v-row>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(5)>

              </v-text-field>
            </v-row>
          </v-col>
          <v-col>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(6)>

              </v-text-field>              
            </v-row>
            <v-row class="rowHWR">
              <v-text-field :label=getStatName(7)>

              </v-text-field>
            </v-row>
          </v-col>
          <v-col id="buttonColHWR">
            <v-row class="rowHWR" id="buttonRowHWR" justify='center'>
              <v-btn id="actionButtonHWR" color="red" rounded="xl">
              GO
              </v-btn>
            </v-row>
          </v-col>          
        </v-row>
      </v-card>
    </v-container>
  </v-app>
</template>

<script>
  export default {
    data: () => ({
      inputStats:
      ["EFG_O",
      "EFG_D",
      "TOR",
      "TORD",
      "ORB",
      "DRB",
      "FTR",
      "FTRD"
      ]
    }),
  methods: {
      getStatName(value) {
        return this.inputStats[value]
      }
    }
  }

</script>

<style>
#baseHWR {
  background-color: burlywood;
}
#cardHWR {
  min-height: 95%;
  padding: auto;
}
#actionButtonHWR {
  padding:auto;
  min-width: 80%;
}
#buttonRowHWR {
  align-items: center;
  top: 50%;
}
#parentRowHWR {
  margin: 0px;
}
.containerHWR {
  margin-top: 10px;
  min-height: 95%;
}
.rowHWR {
  padding: 5px;
  margin-top: 0px !important;
}
</style>