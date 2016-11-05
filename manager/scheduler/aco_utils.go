package scheduler

import (
	"bytes"
	"fmt"
	"math/rand"
	"text/tabwriter"

	"github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/log"
	"github.com/skelterjohn/go.matrix"
)

func arg_max(items []float64) int {
	max_val := -1.0
	max_i := -1
	for i, item := range items {
		if max_val < item {
			max_val = item
			max_i = i
		}
	}
	return max_i
}

func pick(items []float64) int {
	p := rand.Float64()
	cumulativeProbability := 0.0
	for i, item := range items {
		cumulativeProbability += item
		if p <= cumulativeProbability {
			return i
		}
	}
	return 0
}

func shuffleInts(slc []int) {
	N := len(slc)
	for i := 0; i < N; i++ {
		// choose index uniformly in [i, N-1]
		r := i + rand.Intn(N-i)
		slc[r], slc[i] = slc[i], slc[r]
	}
}

func probTran(mat *matrix.DenseMatrix) *matrix.DenseMatrix {
	ret := mat.Copy()
	for i := 0; i < mat.Rows(); i++ {
		sum := 0.0
		for j := 0; j < mat.Cols(); j++ {
			sum += mat.Get(i, j)
		}
		for j := 0; j < mat.Cols(); j++ {
			ret.Set(i, j, mat.Get(i, j)/sum)
		}
	}
	return ret
}

func print(mat *matrix.DenseMatrix) {
	buf := new(bytes.Buffer)
	w := tabwriter.NewWriter(buf, 10, 4, 0, ' ', 0)
	for i := 0; i < mat.Rows(); i++ {
		for j := 0; j < mat.Cols(); j++ {
			fmt.Fprintf(w, "%.6f\t", mat.Get(i, j))
		}
		fmt.Fprint(w, "\n")
	}
	w.Flush()
	log.L.Debugln(buf.String())
}

func resourceReservations(task *api.Task) (reservations api.Resources) {
	if task == nil {
		return
	}

	spec := task.Spec
	if spec.Resources != nil && spec.Resources.Reservations != nil {
		reservations = *spec.Resources.Reservations
	}
	return
}

func availableResources(node NodeInfo) (resource api.Resources) {
	return node.AvailableResources
}
