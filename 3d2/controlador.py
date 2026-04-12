import socket
import ttkbootstrap as tb
from ttkbootstrap.constants import EW, NSEW, W
from ttkbootstrap.scrolled import ScrolledText

UDP_IP = '127.0.0.1'
UDP_PORT = 5005


class UdpSender:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: str) -> None:
        data = (cmd.strip() + '\n').encode('utf-8')
        self.sock.sendto(data, self.addr)


class ControllerApp:
    def __init__(self, root: tb.Window) -> None:
        self.root = root
        self.root.title('Simulation UDP Controller')
        self.root.geometry('1080x860')
        self.sender = UdpSender(UDP_IP, UDP_PORT)
        self.log_var = tb.StringVar(value='Ready')

        self.zoom_var = tb.DoubleVar(value=2.10)
        self.fov_var = tb.DoubleVar(value=14.0)
        self.tilt_var = tb.DoubleVar(value=1.0)
        self.orbit_var = tb.DoubleVar(value=45.0)
        self.exposure_var = tb.DoubleVar(value=1.25)
        self.autoexposure_var = tb.BooleanVar(value=True)
        self.autoreturn_var = tb.BooleanVar(value=False)

        self.log_widget = None
        self._build_ui()
        self._bind_keys()

    def _send(self, cmd: str) -> None:
        self.sender.send(cmd)
        self.log_var.set(f'Sent: {cmd}')
        if self.log_widget is not None:
            self.log_widget.insert('end', f'{cmd}\n')
            self.log_widget.see('end')

    def _build_button(self, parent, text: str, cmd: str, row: int, col: int, colspan: int = 1, sticky: str = NSEW, bootstyle: str = 'secondary-outline'):
        b = tb.Button(parent, text=text, command=lambda c=cmd: self._send(c), bootstyle=bootstyle)
        b.grid(row=row, column=col, columnspan=colspan, padx=4, pady=4, sticky=sticky)
        return b

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = tb.Frame(self.root, padding=14)
        main.grid(sticky=NSEW)
        for i in range(3):
            main.columnconfigure(i, weight=1)

        row = 0
        title = tb.Label(main, text='Renderer UDP Controller', font=('Segoe UI', 18, 'bold'), bootstyle='inverse-primary')
        title.grid(row=row, column=0, columnspan=3, sticky=EW, pady=(0, 10))
        row += 1

        subtitle = tb.Label(main, text=f'Target: {UDP_IP}:{UDP_PORT}', bootstyle='secondary')
        subtitle.grid(row=row, column=0, columnspan=3, sticky=W, pady=(0, 12))
        row += 1

        movement = tb.Labelframe(main, text='Player movement', padding=10, bootstyle='primary')
        movement.grid(row=row, column=0, sticky=NSEW, padx=4, pady=4)
        camera = tb.Labelframe(main, text='Camera and focus', padding=10, bootstyle='info')
        camera.grid(row=row, column=1, sticky=NSEW, padx=4, pady=4)
        render = tb.Labelframe(main, text='Projection and exposure', padding=10, bootstyle='warning')
        render.grid(row=row, column=2, sticky=NSEW, padx=4, pady=4)
        row += 1

        for f in (movement, camera, render):
            for i in range(4):
                f.columnconfigure(i, weight=1)

        self._build_button(movement, 'Up', 'MOVE_UP', 0, 1, bootstyle='success')
        self._build_button(movement, 'Left', 'MOVE_LEFT', 1, 0, bootstyle='success')
        self._build_button(movement, 'Down', 'MOVE_DOWN', 1, 1, bootstyle='success')
        self._build_button(movement, 'Right', 'MOVE_RIGHT', 1, 2, bootstyle='success')
        tb.Label(movement, text='Keyboard: W A S D', bootstyle='secondary').grid(row=2, column=0, columnspan=4, sticky=W, pady=(8, 0))

        self._build_button(camera, 'Orbit left', 'ORBIT_LEFT', 0, 0, bootstyle='info-outline')
        self._build_button(camera, 'Orbit right', 'ORBIT_RIGHT', 0, 1, bootstyle='info-outline')
        self._build_button(camera, 'Tilt up', 'TILT_UP', 0, 2, bootstyle='info-outline')
        self._build_button(camera, 'Tilt down', 'TILT_DOWN', 0, 3, bootstyle='info-outline')
        self._build_button(camera, 'Zoom in', 'ZOOM_IN', 1, 0, bootstyle='info')
        self._build_button(camera, 'Zoom out', 'ZOOM_OUT', 1, 1, bootstyle='info')
        self._build_button(camera, 'Next NPC', 'NEXT_NPC', 1, 2, bootstyle='primary')
        self._build_button(camera, 'Prev NPC', 'PREV_NPC', 1, 3, bootstyle='primary')
        self._build_button(camera, 'Follow player', 'FOLLOW_PLAYER', 2, 0, 2, bootstyle='secondary')
        self._build_button(camera, 'Follow NPC', 'FOLLOW_NPC', 2, 2, 2, bootstyle='secondary')
        self._build_button(camera, 'Toggle follow', 'TOGGLE_FOLLOW', 3, 0, 4, bootstyle='dark')

        self._build_button(render, 'Conic', 'PROJECTION_CONIC', 0, 0, bootstyle='warning')
        self._build_button(render, 'Orthographic', 'PROJECTION_ORTHO', 0, 1, bootstyle='warning')
        self._build_button(render, 'Toggle proj.', 'TOGGLE_PROJECTION', 0, 2, 2, bootstyle='warning-outline')
        self._build_button(render, 'FOV -', 'FOV_DEC', 1, 0, bootstyle='secondary-outline')
        self._build_button(render, 'FOV +', 'FOV_INC', 1, 1, bootstyle='secondary-outline')
        self._build_button(render, 'Exposure -', 'EXPOSURE_DEC', 1, 2, bootstyle='danger-outline')
        self._build_button(render, 'Exposure +', 'EXPOSURE_INC', 1, 3, bootstyle='danger-outline')
        self._build_button(render, 'Autoexp on', 'AUTOEXPOSURE_ON', 2, 0, bootstyle='success-outline')
        self._build_button(render, 'Autoexp off', 'AUTOEXPOSURE_OFF', 2, 1, bootstyle='danger-outline')
        self._build_button(render, 'Toggle autoexp', 'TOGGLE_AUTOEXPOSURE', 2, 2, bootstyle='secondary')
        self._build_button(render, 'Reset exposure', 'EXPOSURE_RESET', 2, 3, bootstyle='dark-outline')

        sliders = tb.Labelframe(main, text='Direct values', padding=10, bootstyle='success')
        sliders.grid(row=row, column=0, columnspan=3, sticky=NSEW, padx=4, pady=8)
        sliders.columnconfigure(1, weight=1)
        row += 1

        self._slider_row(sliders, 0, 'Zoom', self.zoom_var, 0.12, 6.0, self._send_zoom)
        self._slider_row(sliders, 1, 'Orbit degrees', self.orbit_var, -180.0, 180.0, self._send_orbit)
        self._slider_row(sliders, 2, 'Tilt', self.tilt_var, 0.45, 1.65, self._send_tilt)
        self._slider_row(sliders, 3, 'Perspective FOV', self.fov_var, 8.0, 85.0, self._send_fov)
        self._slider_row(sliders, 4, 'Exposure', self.exposure_var, 0.05, 20.0, self._send_exposure)

        toggles = tb.Labelframe(main, text='State toggles', padding=10, bootstyle='secondary')
        toggles.grid(row=row, column=0, columnspan=3, sticky=NSEW, padx=4, pady=4)
        toggles.columnconfigure(0, weight=1)
        toggles.columnconfigure(1, weight=1)
        row += 1

        tb.Checkbutton(toggles, text='Auto exposure', variable=self.autoexposure_var, command=self._toggle_autoexposure, bootstyle='round-toggle').grid(row=0, column=0, sticky=W, padx=4, pady=4)
        tb.Checkbutton(toggles, text='Auto return to player when moving', variable=self.autoreturn_var, command=self._toggle_autoreturn, bootstyle='round-toggle').grid(row=0, column=1, sticky=W, padx=4, pady=4)

        quick = tb.Labelframe(main, text='Quick presets', padding=10, bootstyle='dark')
        quick.grid(row=row, column=0, columnspan=3, sticky=NSEW, padx=4, pady=4)
        row += 1
        for i in range(6):
            quick.columnconfigure(i, weight=1)
        self._build_button(quick, 'NPC cinematic', 'FOLLOW_NPC', 0, 0, bootstyle='primary')
        self._build_button(quick, 'Player camera', 'FOLLOW_PLAYER', 0, 1, bootstyle='secondary')
        self._build_button(quick, 'Conic + NPC', 'PROJECTION_CONIC', 0, 2, bootstyle='warning')
        self._build_button(quick, 'Ortho + NPC', 'PROJECTION_ORTHO', 0, 3, bootstyle='warning-outline')
        self._build_button(quick, 'Wide FOV', 'SET_FOV 40', 0, 4, bootstyle='info')
        self._build_button(quick, 'Tight FOV', 'SET_FOV 14', 0, 5, bootstyle='info-outline')

        status = tb.Label(main, textvariable=self.log_var, anchor='w', bootstyle='inverse-secondary')
        status.grid(row=row, column=0, columnspan=3, sticky=EW, padx=4, pady=(8, 0))
        row += 1

        notes = tb.Label(
            main,
            text='Tip: the renderer can still keep its own keyboard shortcuts. This panel sends the same actions over UDP.',
            justify='left',
            bootstyle='secondary'
        )
        notes.grid(row=row, column=0, columnspan=3, sticky=W, padx=4, pady=(8, 0))
        row += 1

        log_frame = tb.Labelframe(main, text='UDP command log', padding=10, bootstyle='light')
        log_frame.grid(row=row, column=0, columnspan=3, sticky=NSEW, padx=4, pady=(8, 0))
        main.rowconfigure(row, weight=1)
        self.log_widget = ScrolledText(log_frame, height=10, autohide=True)
        self.log_widget.pack(fill='both', expand=True)

    def _slider_row(self, parent, row: int, label: str, variable: tb.DoubleVar, vmin: float, vmax: float, callback) -> None:
        tb.Label(parent, text=label).grid(row=row, column=0, sticky=W, padx=4, pady=4)
        scale = tb.Scale(parent, from_=vmin, to=vmax, variable=variable, command=lambda _=None, cb=callback: cb(), bootstyle='success')
        scale.grid(row=row, column=1, sticky=EW, padx=4, pady=4)
        entry = tb.Entry(parent, width=10)
        entry.grid(row=row, column=2, sticky='e', padx=4, pady=4)
        entry.insert(0, f'{variable.get():.2f}')

        def sync_entry(*_):
            entry.delete(0, 'end')
            entry.insert(0, f'{variable.get():.2f}')
        variable.trace_add('write', sync_entry)

        def apply_entry(_event=None):
            try:
                variable.set(float(entry.get().strip()))
                callback()
            except ValueError:
                sync_entry()
        entry.bind('<Return>', apply_entry)
        tb.Button(parent, text='Send', command=callback, bootstyle='success-outline').grid(row=row, column=3, sticky=EW, padx=4, pady=4)

    def _send_zoom(self) -> None:
        self._send(f'SET_ZOOM {self.zoom_var.get():.4f}')

    def _send_orbit(self) -> None:
        self._send(f'SET_ORBIT_DEG {self.orbit_var.get():.4f}')

    def _send_tilt(self) -> None:
        self._send(f'SET_TILT {self.tilt_var.get():.4f}')

    def _send_fov(self) -> None:
        self._send(f'SET_FOV {self.fov_var.get():.4f}')

    def _send_exposure(self) -> None:
        self.autoexposure_var.set(False)
        self._send('AUTOEXPOSURE_OFF')
        self._send(f'SET_EXPOSURE {self.exposure_var.get():.4f}')

    def _toggle_autoexposure(self) -> None:
        self._send('AUTOEXPOSURE_ON' if self.autoexposure_var.get() else 'AUTOEXPOSURE_OFF')

    def _toggle_autoreturn(self) -> None:
        self._send('AUTO_RETURN_ON' if self.autoreturn_var.get() else 'AUTO_RETURN_OFF')

    def _bind_keys(self) -> None:
        self.root.bind('<Up>', lambda e: self._send('TILT_UP'))
        self.root.bind('<Down>', lambda e: self._send('TILT_DOWN'))
        self.root.bind('<Left>', lambda e: self._send('ORBIT_LEFT'))
        self.root.bind('<Right>', lambda e: self._send('ORBIT_RIGHT'))
        self.root.bind('w', lambda e: self._send('MOVE_UP'))
        self.root.bind('s', lambda e: self._send('MOVE_DOWN'))
        self.root.bind('a', lambda e: self._send('MOVE_LEFT'))
        self.root.bind('d', lambda e: self._send('MOVE_RIGHT'))
        self.root.bind('<Prior>', lambda e: self._send('ZOOM_IN'))
        self.root.bind('<Next>', lambda e: self._send('ZOOM_OUT'))
        self.root.bind('n', lambda e: self._send('NEXT_NPC'))
        self.root.bind('b', lambda e: self._send('PREV_NPC'))
        self.root.bind('p', lambda e: self._send('TOGGLE_PROJECTION'))
        self.root.bind('f', lambda e: self._send('TOGGLE_FOLLOW'))
        self.root.bind('e', lambda e: self._send('TOGGLE_AUTOEXPOSURE'))


def main() -> None:
    root = tb.Window(themename='darkly')
    ControllerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
