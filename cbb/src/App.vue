<script setup>
import { ref, computed } from 'vue';
import HWRPage from './components/HWRPage.vue';
import WinRate from './components/WinRatePredictorPage.vue';
import Home from './components/HomePage.vue';
import Evaluation from './components/Eval.vue'
import Weights from './components/Weights.vue';
// const routes = {
//   '/': Home,
//   '/historical-win-rate': HWRPage,
//   '/win-rate-predictor': WinRate,
// }

// const currentPath = ref(window.location.hash)

// window.addEventListener('hashchange', () => {
//   currentPath.value = window.location.hash
// })

// const currentView = computed(() => {
//   console.log(routes)
//   return routes[1]
// })
// </script>

<template id = "base">
    <v-app>
      <v-card>
        <v-tabs align-tabs="center">
          <v-tab :value="1" @click=changeComponent(0)>Feature Comparison</v-tab>
          <v-tab :value="2" @click=changeComponent(1)>Win Rate Predictor</v-tab>
          <v-tab :value="3" @click=changeComponent(2)>Historical Win Rate</v-tab>
          <v-tab :value="4" @click=changeComponent(3)>Model Evaluation</v-tab>
          <v-tab :value="5" @click=changeComponent(4)>Weight Interpretations</v-tab>
        </v-tabs>
      </v-card>
      <main id = 'body'>
        <component :is="activeComponent"/>
      </main>
    </v-app>
</template>

<script>
  export default {
    data: () => ({
      currentIDX: 0,
      componentList: [
        Home, WinRate, HWRPage, Evaluation, Weights
      ],
      activeComponent: Home
    }),

    methods: {
      changeComponent(value) {
        this.currentIDX = value
        this.activeComponent = this.componentList[this.currentIDX]
      }
    }
  }
</script>

<style>
  #body {
    min-height: 95%;
    background-color: burlywood;
  }
  #page1 {
    min-height: 800px;
    min-width: 800px;
    display: block;
  }
</style>