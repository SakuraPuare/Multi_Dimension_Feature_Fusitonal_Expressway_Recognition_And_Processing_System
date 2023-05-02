<template>
    <el-radio-group v-model="direction">
        <el-radio label="ltr">left to right</el-radio>

    </el-radio-group>

    <el-drawer
            v-model="drawer"
            :before-close="handleClose"
            direction="ltr"
            title="I am the title"
    >
        <span>Hi, there!</span>
    </el-drawer>
    <el-drawer v-model="drawer2" :direction="direction">
        <template #header>
            <h4>set title by slot</h4>
        </template>
        <template #default>
            <div>
                <el-radio v-model="radio1" label="Option 1" size="large"
                >Option 1
                </el-radio
                >
                <el-radio v-model="radio1" label="Option 2" size="large"
                >Option 2
                </el-radio
                >
            </div>
        </template>
        <template #footer>
            <div style="flex: auto">
                <el-button @click="cancelClick">cancel</el-button>
                <el-button type="primary" @click="confirmClick">confirm</el-button>
            </div>
        </template>
    </el-drawer>
</template>

<script setup>
import {ref} from 'vue'
import {ElMessageBox} from 'element-plus'


const drawer = ref(false)
const drawer2 = ref(false)
const direction = ref('rtl')
const radio1 = ref('Option 1')


const props = defineProps({
    show_drawer: {
        type: Boolean,
        default: false
    }
})
const handleClose = () => {
    ElMessageBox.confirm('Are you sure you want to close this?')
        .then(() => {
            done()
        })
        .catch(() => {
            // catch error
        })
}

function cancelClick() {
    drawer2.value = false
}

function confirmClick() {
    ElMessageBox.confirm(`Are you confirm to chose ${radio1.value} ?`)
        .then(() => {
            drawer2.value = false
        })
        .catch(() => {
            // catch error
        })
}
</script>
